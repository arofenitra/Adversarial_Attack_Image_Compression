############################################
# Algorithms for AI image compression attack
#
# ANH-HUY PHAN
############################################

import math
import os
import pickle
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import torch.nn.functional as F
from kornia.losses import PSNRLoss, SSIMLoss
from PIL import Image

# for binary mask
from skimage.morphology import dilation, erosion, opening, square
from torchvision import transforms


# Define a wrapper function for SSIMLoss
def ssim(input, target, window_size=11):
    ssim_loss = SSIMLoss(window_size=window_size)
    return ssim_loss(input, target)


def dists(input, target):
    dists_loss = piq.DISTS(reduction="none")
    return dists_loss(input, target)


def bpp_loss_0(output, num_pixels):
    bpp = (
        torch.log(output["likelihoods"]["y"]).sum()
        + torch.log(output["likelihoods"]["z"]).sum()
    ) / (-math.log(2) * num_pixels)
    return bpp


def bpp_loss(output, num_pixels):
    bpp = sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in output["likelihoods"].values()
    )
    return bpp


def export_results_to_image(
    perturbed_image, perturbed_output, noise_pattern, file_name, methodname
):
    file_no_extension = os.path.splitext(file_name)[0]

    perturbed_image_arr = (
        perturbed_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    )

    # ensure the values are in the range [0, 1]
    perturbed_image_arr = np.clip(perturbed_image_arr, 0, 1)

    # scale the values to [0, 255] and convert to uint8
    perturbed_image_arr = (perturbed_image_arr * 255).astype(np.uint8)

    # Create a PIL Image and save it
    im = Image.fromarray(perturbed_image_arr)
    file_name = file_no_extension + "_perturbed_" + methodname + ".png"
    im.save(file_name)

    # ----
    perturbed_output_arr = (
        perturbed_output.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    )

    # ensure the values are in the range [0, 1]
    perturbed_output_arr = np.clip(perturbed_output_arr, 0, 1)

    # scale the values to [0, 255] and convert to uint8
    perturbed_output_arr = (perturbed_output_arr * 255).astype(np.uint8)

    # Create a PIL Image and save it
    im = Image.fromarray(perturbed_output_arr)
    file_name = file_no_extension + "_decompress_" + methodname + ".png"

    im.save(file_name)

    # ----
    noise_pattern_arr = (
        noise_pattern.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    )

    # normalize shift the noise to 0-1
    minn = np.min(noise_pattern_arr.ravel())
    maxn = np.max(noise_pattern_arr.ravel())
    noise_pattern_arr = (noise_pattern_arr - minn) / (maxn - minn)

    # # ensure the values are in the range [0, 1]
    # noise_pattern_arr = np.clip(noise_pattern_arr, 0, 1)

    # scale the values to [0, 255] and convert to uint8
    noise_pattern_arr = (noise_pattern_arr * 255).astype(np.uint8)

    # Create a PIL Image and save it
    im = Image.fromarray(noise_pattern_arr)
    file_name = file_no_extension + "_noise_" + methodname + ".png"

    im.save(file_name)

    return


# -----for saving latents to file


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def savecompressed(compressfile, outnet, bitdepth, h, w):
    # with torch.no_grad():
    # outnet = net.compress(image)

    shape = outnet["shape"]

    with Path(compressfile).open("wb") as f:
        # write_uchars(f, codec.codec_header)
        # write original image size
        write_uints(f, (h, w))
        # write original bitdepth
        write_uchars(f, (bitdepth,))
        # write shape and number of encoded latents
        write_body(f, shape, outnet["strings"])

    size = filesize(compressfile)
    bpp = float(size) * 8 / (h * w)

    return bpp


def gaussian_kernel(size, sigma):
    # Create a vector of size 'size' with values from -size//2 to size//2
    x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
    # Calculate the Gaussian distribution for each value in the vector
    g = torch.exp(-(x**2) / (2 * sigma**2))
    # Normalize the distribution so it sums to 1
    g /= g.sum()
    # Create a 2D Gaussian kernel from the outer product of the vector with itself
    return g.outer(g)

def gaussian_filter_mask(x,device):
    x = x.to(device)
    # Define the size and standard deviation of the Gaussian kernel
    kernel_size = 5
    sigma = 1.0

    # Create the Gaussian kernel
    gaussian_filter = gaussian_kernel(kernel_size, sigma)

    # Add batch and channel dimensions to the filter
    gaussian_filter = gaussian_filter.view(1, 1, *gaussian_filter.size())

    # Assuming 'image' is with shape [batch_size, channels, height, width]
    # Repeat the filter for each input channel
    gaussian_filter = gaussian_filter.repeat(x.size(1), 1, 1, 1)
    gaussian_filter = gaussian_filter.to(device)
    

    # Gaussian blur to the mask to smooth the edges
    kernel_size = 21
    sigma = 8

    # Create the Gaussian kernel
    gaussian_filter_mask = gaussian_kernel(kernel_size, sigma)

    # Add batch and channel dimensions to the filter
    gaussian_filter_mask = gaussian_filter_mask.view(1, 1, *gaussian_filter_mask.size())

    # Assuming 'image' is with shape [batch_size, channels, height, width]
    # Repeat the filter for each input channel
    gaussian_filter_mask = gaussian_filter_mask.repeat(x.size(1), 1, 1, 1)
    gaussian_filter_mask = gaussian_filter_mask.to(device)

    return gaussian_filter_mask
##___________________________________________#######

def addition_noise_torch(image_tensor, noise_tensor, device,g):
    x = image_tensor.to(device)
    e = noise_tensor.to(device)
    if g == 'logexp':
        return torch.log(torch.exp(x) + e).to(device)
    elif g == 'tanhatanh':
        return torch.tanh(torch.arctanh(x) + e).to(device)
    elif g == 'logexpt':
        return torch.log(torch.exp(x) + torch.tanh(e)).to(device)
    elif g == 'tanhatanht':
        return torch.tanh(torch.arctanh(x) + torch.tanh(e)).to(device)
    elif g == 'addition':
        return (x+e).to(device)
    elif g == 'additiont':
        return (x+torch.tanh(e)).to(device)
    else:
        return x.to(device)

def new_maxdistortion(
    x,
    errbound=0.1,
    smoothfilter=None,
    losstype="psnr",
    l1_lambda=0,
    num_iterations=1000,
    model=None,
    device=None,
    mask=None,
    initial_noise=None,
    learningrate=0.1,
    g = "tanharctanh"
):
    """
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    """
    #
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    if learningrate is None:
        learningrate = 0.1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            # noise_pattern.data.clamp_(-errbound, errbound)
            if g[-1]=="t":
                noise_pattern2 = errbound * torch.tanh(noise_pattern)
            else:
                noise_pattern2 = errbound * noise_pattern

            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern2 * mask
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(
                    noise_pattern2 * mask,
                    smoothfilter,
                    padding=kernel_size // 2,
                    groups=x.size(1),
                )

            # Apply current noise pattern
            if g[-1]!="t":
                perturbed_image = addition_noise_torch(x,smoothed_noise_pattern, device,g)
            else:
                perturbed_image = addition_noise_torch(x,smoothed_noise_pattern, device,g[:-1])

            # Forward pass through the model
            output = model(perturbed_image)

            # output['x_hat'] for the reconstructed image
            perturbed_output = output["x_hat"]

            if losstype == "psnr":
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I**2) / mse_loss)

                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR

            elif losstype == "ssim":
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)

                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM

            elif losstype == "dists":

                dists_perturbed = dists(x, perturbed_output)
                loss = -dists_perturbed

            if l1_lambda > 0:
                # L1-norm for sparsity
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}")

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint(
            {
                "noise_pattern": noise_pattern.data,
                "smoothed_noise_pattern": smoothed_noise_pattern.data,
                "perturbed_image": perturbed_image.data,
                "perturbed_output": perturbed_output.data,
                "iteration": iteration,
            }
        )
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern 


def new_maxbpp(x, errbound=0.1, smoothfilter=None, qualitymeasure='psnr',
               target_quality=None,quality_loss_lambda=0.1,l1_lambda=0,
               num_iterations=1000, model=None, device=None, mask=None,
               initial_noise=None,learningrate=1,g="tanhatanh"):
    
    # Attack the whole image with a noise pattern which 
    # - maximizes the loss of the compression performance : maximize bpp 
    # - preserves the PSNR of the decompressed image :   min |PSNR(f(x + n)) - PSNR(f(x))|
    # - Sparse and smooth perturbed noise
    '''
        min_n -bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
        s.t  |n_{i,kj}|<= sigma
    '''
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer 
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1
        
    if learningrate is None:
        learningrate = 1

    if target_quality is None:
        target_quality = 0

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(device)
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]


    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            #noise_pattern.data.clamp_(-errbound, errbound)
            if g[-1]=="t":
                noise_pattern2 = errbound * torch.tanh(noise_pattern)
            else:
                noise_pattern2 = errbound * noise_pattern

            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern2 * mask
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(
                    noise_pattern2 * mask,
                    smoothfilter,
                    padding=kernel_size // 2,
                    groups=x.size(1),
                )

            # Apply current noise pattern
            if g[-1]!="t":
                perturbed_image = addition_noise_torch(x,smoothed_noise_pattern, device,g)
            else:
                perturbed_image = addition_noise_torch(x,smoothed_noise_pattern, device,g[:-1])


            # Forward pass through the model
            output = model(perturbed_image)

            # Assuming 'output' is a dictionary with key 'x_hat' for the reconstructed image
            perturbed_output = output['x_hat']

            if qualitymeasure == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                perturbed_quality = 10 * torch.log10((MAX_I ** 2) / mse_loss)

                # Compute the difference in PSNR between perturbed and target
                quality_loss = (perturbed_quality - target_quality).abs()


            elif qualitymeasure == 'ssim':
                # maximize distortion = minimize 1-SSIM
                perturbed_quality = ssim(perturbed_output, x)

                quality_loss = (perturbed_quality - target_quality).abs() 


            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                
            else:
                l1norm = 0

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # quality_loss_lambda = 0.1
            
            # Combine the losses
            # combined_loss = -bpploss + quality_loss_lambda * quality_loss + l1_lambda * l1norm
    
            combined_loss = -10*torch.log10(bpploss) + quality_loss_lambda * torch.log10(quality_loss) + l1_lambda * l1norm


            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                
                print(f'Iteration {iteration} | {qualitymeasure}: {perturbed_quality} - Lost {quality_loss} | BPP {bpploss} |  Loss {combined_loss}')
                
    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': noise_pattern.data, \
                        'smoothed_noise_pattern': smoothed_noise_pattern.data,\
                        'perturbed_image': perturbed_image.data,\
                        'perturbed_output': perturbed_output.data,\
                        'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern
    
    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

def quantization_noise_mask(noise_level,smoothed_noise_pattern):
    quantized_smoothed_noise_pattern = smoothed_noise_pattern.clone()
    quantized_smoothed_noise_pattern[smoothed_noise_pattern<-0.01] = -noise_level
    quantized_smoothed_noise_pattern[smoothed_noise_pattern>0.01] = noise_level
    quantized_smoothed_noise_pattern[(smoothed_noise_pattern<=0.01)*(smoothed_noise_pattern>=-0.01)] = 0
    quantized_perturbed_image = x + quantized_smoothed_noise_pattern
    return quantized_perturbed_image
    
def maxdistortion_tanh(
    x,
    errbound=0.1,
    smoothfilter=None,
    losstype="psnr",
    l1_lambda=0,
    num_iterations=1000,
    model=None,
    device=None,
    mask=None,
    initial_noise=None,
    learningrate=0.1,
):
    """
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    """
    #
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    if learningrate is None:
        learningrate = 0.1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            # noise_pattern.data.clamp_(-errbound, errbound)
            noise_pattern2 = errbound * torch.tanh(noise_pattern)

            # Smooth the noise pattern
            if smoothfilter is None:
                smoothed_noise_pattern = noise_pattern2 * mask
            else:
                kernel_size = smoothfilter.shape[-1]
                smoothed_noise_pattern = F.conv2d(
                    noise_pattern2 * mask,
                    smoothfilter,
                    padding=kernel_size // 2,
                    groups=x.size(1),
                )

            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern

            # Forward pass through the model
            output = model(perturbed_image)

            # output['x_hat'] for the reconstructed image
            perturbed_output = output["x_hat"]

            if losstype == "psnr":
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I**2) / mse_loss)

                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR

            elif losstype == "ssim":
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)

                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM

            elif losstype == "dists":

                dists_perturbed = dists(x, perturbed_output)
                loss = -dists_perturbed

            if l1_lambda > 0:
                # L1-norm for sparsity
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}")

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint(
            {
                "noise_pattern": noise_pattern.data,
                "smoothed_noise_pattern": smoothed_noise_pattern.data,
                "perturbed_image": perturbed_image.data,
                "perturbed_output": perturbed_output.data,
                "iteration": iteration,
            }
        )
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


##___________________________________________#######


def maxdistortion(
    x,
    errbound,
    smoothfilter,
    losstype,
    l1_lambda,
    num_iterations,
    model,
    device=None,
    mask=None,
    initial_noise=None,
):
    """
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    """
    #
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=0.1)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            noise_pattern.data.clamp_(-errbound, errbound)

            # Smooth the noise pattern
            kernel_size = smoothfilter.shape[-1]
            smoothed_noise_pattern = F.conv2d(
                noise_pattern * mask,
                smoothfilter,
                padding=kernel_size // 2,
                groups=x.size(1),
            )

            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern

            # Forward pass through the model
            output = model(perturbed_image)

            # output['x_hat'] for the reconstructed image
            perturbed_output = output["x_hat"]

            if losstype == "psnr":
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I**2) / mse_loss)

                # maximize distortion = minimize PSNR
                loss = psnr_loss  # Negative sign because we want to maximize PSNR

            elif losstype == "ssim":
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)

                loss = -ssim_perturbed  # Negative sign because we want to maximize SSIM

            elif losstype == "dists":

                dists_perturbed = dists(x, perturbed_output)
                loss = -dists_perturbed

            if l1_lambda > 0:
                # L1-norm for sparsity
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}")

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint(
            {
                "noise_pattern": noise_pattern.data,
                "smoothed_noise_pattern": smoothed_noise_pattern.data,
                "perturbed_image": perturbed_image.data,
                "perturbed_output": perturbed_output.data,
                "iteration": iteration,
            }
        )
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


###_______________________________________####


def maxbitrate_v0(
    x,
    errbound,
    smoothfilter,
    qualitymeasure,
    target_quality,
    quality_loss_lambda,
    l1_lambda,
    num_iterations,
    model,
    device=None,
    mask=None,
    initial_noise=None,
):
    # Attack the whole image with a noise pattern which
    # - maximizes the loss of the compression performance : maximize bpp
    # - preserves the PSNR of the decompressed image :   min |PSNR(f(x + n)) - PSNR(f(x))|
    # - Sparse and smooth perturbed noise
    """
    min_n -bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
    s.t  |n_{i,kj}|<= sigma
    """
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=1)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Clamp the noise pattern values to ensure they stay within a valid range
        noise_pattern.data.clamp_(-errbound, errbound)

        # Smooth the noise pattern
        kernel_size = smoothfilter.shape[-1]
        smoothed_noise_pattern = F.conv2d(
            noise_pattern * mask,
            smoothfilter,
            padding=kernel_size // 2,
            groups=x.size(1),
        )

        # Apply current noise pattern
        perturbed_image = x + smoothed_noise_pattern

        # Forward pass through the model
        output = model(perturbed_image)

        # Assuming 'output' is a dictionary with key 'x_hat' for the reconstructed image
        perturbed_output = output["x_hat"]

        if qualitymeasure == "psnr":
            # Calculate MSE loss
            mse_loss = F.mse_loss(perturbed_output, x)

            # Calculate PSNR loss
            perturbed_quality = 10 * torch.log10((MAX_I**2) / mse_loss)

            # Compute the difference in PSNR between perturbed and target
            quality_loss = (perturbed_quality - target_quality).abs()

        elif qualitymeasure == "ssim":
            # maximize distortion = minimize 1-SSIM
            perturbed_quality = ssim(perturbed_output, x)

            quality_loss = (perturbed_quality - target_quality).abs()

        if l1_lambda > 0:
            # L1-norm for sparsity
            l1norm = smoothed_noise_pattern.abs().sum()

        else:
            l1norm = 0

        # Compute the bpp loss
        bpploss = bpp_loss(output, num_pixels)

        # quality_loss_lambda = 0.1

        # Combine the losses
        # combined_loss = -bpploss + quality_loss_lambda * quality_loss + l1_lambda * l1norm

        combined_loss = (
            -10 * torch.log10(bpploss)
            + quality_loss_lambda * torch.log10(quality_loss)
            + +l1_lambda * l1norm
        )

        # Perform gradient descent
        combined_loss.backward()
        optimizer.step()

        # Print the loss every 100 iterations
        if iteration % 100 == 0:

            print(
                f"Iteration {iteration} | {qualitymeasure}: {perturbed_quality} - Lost {quality_loss} | BPP {bpploss} |  Loss {combined_loss}"
            )

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


def noise_to_mask(smoothed_noise_pattern):
    # Define a mask based on the sparse noise pattern
    ell2_noise = torch.sqrt(torch.sum(smoothed_noise_pattern**2, dim=1))

    mask_noise = ell2_noise > torch.max(ell2_noise) * 1e-1
    mask_noise = mask_noise.squeeze()

    mask_noise = erosion(mask_noise.cpu().detach().numpy(), square(5))
    mask_noise = opening(mask_noise, square(5))
    mask_noise = dilation(mask_noise, square(5))
    mask_noise = dilation(mask_noise, square(5))

    # plt.imshow(mask_noise)

    # Mask 3D
    mask = torch.zeros_like(smoothed_noise_pattern)
    nnz_ix = np.where(mask_noise == 1)
    mask[:, :, nnz_ix[0], nnz_ix[1]] = 1

    return mask, mask_noise


# Function to save state
def save_checkpoint(state, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(state, f)


# Function to load state
def load_checkpoint(filename="checkpoint.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    return None


def maxbitrate(
    x,
    errbound,
    smoothfilter,
    qualitymeasure,
    target_quality,
    quality_loss_lambda,
    l1_lambda,
    num_iterations,
    model,
    device=None,
    mask=None,
    initial_noise=None,
):
    # Attack the whole image with a noise pattern which
    # - maximizes the loss of the compression performance : maximize bpp
    # - preserves the PSNR of the decompressed image :   min |PSNR(f(x + n)) - PSNR(f(x))|
    # - Sparse and smooth perturbed noise
    """
    min_n -bpp(theta|n) + l_1 |PSNR(f(x + n)) - PSNR(f(x))| + ll_sparse ||x+n||
    s.t  |n_{i,kj}|<= sigma
    """
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # qualitymeasure: type of quality metric to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # quality_loss_lambda :  weight for quality loss regularizer
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1

    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(
            device
        )
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=1)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Clamp the noise pattern values to ensure they stay within a valid range
            noise_pattern.data.clamp_(-errbound, errbound)

            # Smooth the noise pattern
            kernel_size = smoothfilter.shape[-1]
            smoothed_noise_pattern = F.conv2d(
                noise_pattern * mask,
                smoothfilter,
                padding=kernel_size // 2,
                groups=x.size(1),
            )

            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern

            # Forward pass through the model
            output = model(perturbed_image)

            # Assuming 'output' is a dictionary with key 'x_hat' for the reconstructed image
            perturbed_output = output["x_hat"]

            if qualitymeasure == "psnr":
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, x)

                # Calculate PSNR loss
                perturbed_quality = 10 * torch.log10((MAX_I**2) / mse_loss)

                # Compute the difference in PSNR between perturbed and target
                quality_loss = (perturbed_quality - target_quality).abs()

            elif qualitymeasure == "ssim":
                # maximize distortion = minimize 1-SSIM
                perturbed_quality = ssim(perturbed_output, x)

                quality_loss = (perturbed_quality - target_quality).abs()

            if l1_lambda > 0:
                # L1-norm for sparsity
                l1norm = smoothed_noise_pattern.abs().sum()

            else:
                l1norm = 0

            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)

            # quality_loss_lambda = 0.1

            # Combine the losses
            # combined_loss = -bpploss + quality_loss_lambda * quality_loss + l1_lambda * l1norm

            combined_loss = (
                -10 * torch.log10(bpploss)
                + quality_loss_lambda * torch.log10(quality_loss)
                + +l1_lambda * l1norm
            )

            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iteration % 100 == 0:

                print(
                    f"Iteration {iteration} | {qualitymeasure}: {perturbed_quality} - Lost {quality_loss} | BPP {bpploss} |  Loss {combined_loss}"
                )

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint(
            {
                "noise_pattern": noise_pattern.data,
                "smoothed_noise_pattern": smoothed_noise_pattern.data,
                "perturbed_image": perturbed_image.data,
                "perturbed_output": perturbed_output.data,
                "iteration": iteration,
            }
        )
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern


import copy

# Entropy filtering
from scipy.stats import entropy
from skimage import color, io, segmentation
from skimage.filters.rank import entropy as entropy_filter
from skimage.measure import regionprops
from skimage.morphology import disk

###_________________________________####
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float


def noisemask_maxentropy_superpixel(
    img_rgb, n_segments=400, sigma=4, n_topentropysegments=5, verbose=False
):
    # Return the mask for the selected region of the images (superpixels) which have the highest entropy
    #
    # SLIC super-pixel
    segments = slic(img_rgb, n_segments=n_segments, sigma=sigma)
    imsize_ = img_rgb.shape
    #
    regions = regionprops(segments)

    # Convert to grayscale as entropy is typically computed on grayscale images
    gray_img = color.rgb2gray(img_rgb)

    # Compute histogram for each superpixel
    histograms = []
    for i, segVal in enumerate(np.unique(segments)):
        # Mask the superpixel
        mask = np.zeros(gray_img.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255

        # Compute the histogram
        hist, _ = np.histogram(gray_img[mask > 0], bins=256, range=(0, 1))
        histograms.append(hist)

    # Compute the entropy for each histogram
    entropies = [entropy(hist) for hist in histograms]

    # Select superpixels with high entropy
    # threshold = np.mean(entropies)
    # low_entropy_superpixels = [i for i, e in enumerate(entropies) if e < threshold]
    superpixel_order = np.argsort(entropies)

    # select superpixel with highest entropy
    # superpixel_order: index list of superpixels ordered in the ascending of their entropy
    top_entropy_superpixels = superpixel_order[-n_topentropysegments:]

    # Binary mask for the selected superpixel
    # mask = torch.zeros_like([1 img_rgbx)
    mask = torch.zeros([1, imsize_[2], imsize_[0], imsize_[1]])
    # Set ones at the specified coordinates in the mask
    for segVal in top_entropy_superpixels:
        coord = regions[segVal]["coords"]
        mask[:, :, coord[:, 0], coord[:, 1]] = 1

    # Initialize an array to hold the maximum entropy value for each superpixel
    mean_entropy_segment_img = np.zeros_like(gray_img)
    # Iterate over each unique segment label
    for i, segVal in enumerate(np.unique(segments)):
        # Create a mask for the current segment
        mask_k = segments == segVal

        # Assign this mean value to the corresponding pixels in the output array
        mean_entropy_segment_img[mask_k] = entropies[i]

    if verbose == True:
        # display maps of superpixels
        plt.figure(1)

        # Overlay the segment boundaries in yellow
        minn = np.min(mean_entropy_segment_img.reshape(-1))
        maxn = np.max(mean_entropy_segment_img.reshape(-1))

        mm = (mean_entropy_segment_img - minn) / (maxn - minn)
        plt.imshow(mm)
        marked = mark_boundaries(mm, segments, color=(1, 1, 0))
        # Display the boundaries on top of the colored image
        plt.imshow(marked, alpha=0.2, cmap="gray", interpolation="none")

        # # Show the plot
        plt.show()

        # display maps of superpixels
        plt.figure(2)
        plt.imshow(mask.squeeze().cpu().detach().numpy().transpose(1, 2, 0))

    return mask, superpixel_order, regions, segments, mean_entropy_segment_img


def noisemask_entropyfilter(
    img_rgb, n_segments=400, sigma=4, n_topentropysegments=5, verbose=False
):
    # Return the mask for the selected region of the images (superpixels) which have the highest entropy
    #
    imsize_ = img_rgb.shape
    gray_img = color.rgb2gray(img_rgb)

    # Entropy image
    entr_img = entropy_filter(gray_img, disk(5))

    # Superpixel
    segments = slic(img_rgb, n_segments=n_segments, sigma=sigma)
    regions = regionprops(segments)

    # Initialize an array to hold the maximum entropy value for each superpixel
    mean_entropy_segment_img = np.zeros_like(entr_img)
    mean_entropy_segment = np.zeros(len(np.unique(segments)))
    # Iterate over each unique segment label
    for k in np.unique(segments):
        # Create a mask for the current segment
        mask_k = segments == k

        # Get all entropy values in the current segment
        segment_entropy_values = entr_img[mask_k]

        # Calculate the mean of the top 20 entropy values
        mean_entropy = np.mean(segment_entropy_values)

        # Assign this mean value to the corresponding pixels in the output array
        mean_entropy_segment_img[mask_k] = mean_entropy
        mean_entropy_segment[k - 1] = mean_entropy

    # low_entropy_superpixels = [i for i, e in enumerate(entropies) if e < threshold]
    superpixel_order = np.argsort(mean_entropy_segment)

    # superpixel_order: index list of superpixels ordered in the ascending of their entropy
    top_entropy_superpixels = superpixel_order[-n_topentropysegments:]

    # Binary mask for the selected superpixel
    mask = torch.zeros([1, imsize_[2], imsize_[0], imsize_[1]])
    # Set ones at the specified coordinates in the mask
    for segVal in top_entropy_superpixels:
        coord = regions[segVal]["coords"]
        mask[:, :, coord[:, 0], coord[:, 1]] = 1

    if verbose == True:
        # display maps of superpixels
        plt.figure(1)

        # Overlay the segment boundaries in yellow
        minn = np.min(mean_entropy_segment_img.reshape(-1))
        maxn = np.max(mean_entropy_segment_img.reshape(-1))

        mm = (mean_entropy_segment_img - minn) / (maxn - minn)
        plt.imshow(mm)
        marked = mark_boundaries(mm, segments, color=(1, 1, 0))
        # Display the boundaries on top of the colored image
        plt.imshow(marked, alpha=0.2, cmap="gray", interpolation="none")

        # # Show the plot
        plt.show()

        # display maps of superpixels
        plt.figure(2)
        plt.imshow(mask.squeeze().cpu().detach().numpy().transpose(1, 2, 0))

    return mask, superpixel_order, regions, segments, mean_entropy_segment_img


###_______________________________________###
# Calculate baseline peformance for the original output
def eval_perf(model, net, x, img_path):
    MAX_I = 1
    num_pixels = x.shape[2] * x.shape[3]
    # model and net are the same , but net is on cpu and for compression
    with torch.no_grad():
        original_output = model(x)
        mse_loss_original = F.mse_loss(original_output["x_hat"].clamp_(0, 1), x)
        target_psnr = 10 * torch.log10((MAX_I**2) / mse_loss_original)

        # Compute bpp loss (to be maximized, hence the negative sign)
        baseline_bpp = (
            torch.log(original_output["likelihoods"]["y"]).sum()
            + torch.log(original_output["likelihoods"]["z"]).sum()
        ) / (-math.log(2) * num_pixels)

        target_ssim = ssim(original_output["x_hat"].clamp_(0, 1), x)

        original_output = net.compress(x.to("cpu"))

    # Generate a random number
    unique_id = np.random.randint(1000, 9999)

    # Create the new filename with 'compress' and the unique_id
    compressfile = os.path.splitext(img_path)[0] + "compress" + str(unique_id)

    bitdepth = 8
    h, w = x.size(2), x.size(3)
    baseline_true_bpp = savecompressed(compressfile, original_output, bitdepth, h, w)

    result = {
        "PSNR": target_psnr.cpu().detach().numpy(),
        "Bpp": baseline_bpp.cpu().detach().numpy(),
        "Bpp(fsize)": baseline_true_bpp,
        "SSIM": target_ssim.cpu().detach().numpy(),
    }
    return result


def vis_results(perturbed_image, perturbed_output, smoothed_noise_pattern):
    plt.figure(1)
    plt.imshow(perturbed_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.show
    plt.figure(2)
    plt.imshow(perturbed_output.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.show
    plt.figure(3)
    minn = np.min(
        smoothed_noise_pattern.squeeze()
        .cpu()
        .detach()
        .numpy()
        .transpose(1, 2, 0)
        .ravel()
    )
    maxn = np.max(
        smoothed_noise_pattern.squeeze()
        .cpu()
        .detach()
        .numpy()
        .transpose(1, 2, 0)
        .ravel()
    )
    plt.imshow(
        (
            smoothed_noise_pattern.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
            - minn
        )
        / (maxn - minn)
    )
    plt.axis("off")
    plt.show

    return


import pandas as pd


# Function to check if two dictionaries have the same structure
def have_same_structure(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if type(dict1[key]) != type(dict2[key]):
            return False
    return True


# Function to collect performance data into a DataFrame
def collect_perf(all_vars, template_var_name):
    # Find variables with the same structure as 'template'
    similar_structures = [
        name
        for name, var in all_vars.items()
        if isinstance(var, dict)
        and have_same_structure(all_vars[template_var_name], var)
    ]

    # Create a DataFrame from the dictionaries
    data = {"Method": [], "PSNR": [], "Bpp": [], "Bpp(fsize)": [], "SSIM": []}

    # Populate the DataFrame
    for var_name in similar_structures:
        var = all_vars[var_name]
        data["Method"].append(var_name)
        data["PSNR"].append(var["PSNR"])
        data["Bpp"].append(var["Bpp"])
        data["Bpp(fsize)"].append(var["Bpp(fsize)"])
        data["SSIM"].append(var["SSIM"])

    # Convert the dictionary to a DataFrame and sort it
    df = pd.DataFrame(data)
    df = df.sort_values("Bpp")

    return df


# # Usage example:
# # Assuming 'baseline_' is the name of one of the dictionaries and is defined in your current scope
# all_vars = vars().copy()
# df = collect_perf(all_vars, 'baseline_')
# print(df)


def psnr_torch(img1,img2):
    #img1,img2 are tensor image
    se = torch.pow(img1-img2,2)
    mse = se.mean()
    psnr = -10*torch.log10(mse)
    return psnr


def eval_perf_full(model, net, y, x,img_path):

    # x and y are tensor
    #y is original image, x is attacked input
    with torch.no_grad():
        compressed = model(x)
        decompressed = compressed["x_hat"]
    ai = x
    oi = y
    ao = decompressed 
    
    
    # Convert tensors to numpy arrays
    original_img = y.squeeze().cpu().numpy().transpose(1, 2, 0)
    reconstructed_img = decompressed.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Calculate PSNR
    psnr_aooi = psnr_torch(ao, oi)

    # Calculate SSIM using kornia's SSIMLoss
    ssim_loss = SSIMLoss(window_size=11, reduction='mean')
    ssim_value = 1 - ssim_loss(ao,oi).item()

    # Calculate Bpp
    num_pixels = original_img.shape[0] * original_img.shape[1]
    bpp = bpp_loss(compressed,num_pixels)

    # Calculate Bpp based on file size
    fsize = os.path.getsize(img_path)
    bpp_fsize = fsize / num_pixels

    # Extract the reconstructed image tensor
    if torch.equal(ai,oi):
        psnr_aioi = torch.inf
    else:
        psnr_aioi = psnr_torch(ai,oi)

    # Assuming vif is a function that takes tensors
    vif_instance = VisualInformationFidelity().to(device)
    vif_ai_oi = vif_instance(ai, oi)
    vif_ao_oi = vif_instance(ao, oi)

    # Return the results as a dictionary
    return {
        "PSNR(ai,ao)": psnr_aooi,
        'PSNR(ai,oi)': psnr_aioi,
        "Bpp": bpp,
        "Bpp(fsize)": bpp_fsize,
        "SSIM": ssim_value,
        "VIF(ai,oi)": vif_ai_oi,
        "VIF(ao,oi)": vif_ao_oi
    }

#### Defense

def defend_maxdistortion_tanh(x, errbound, smoothfilter, losstype, l1_lambda, num_iterations, model, device=None, mask=None,initial_noise=None,learningrate = 0.1):
    '''
    \min_{n}  PSNR(x_{out} - x_{in}) + \lambda  ||n||_1,     
        s.t.   x_{out} = f(x_{in} + n), |n_{ij}|<= \sigma
    ''' 
    #    This depense works for other attack model
    #
    # x: input image of size 1 x C x H x W
    # errbound: noise bound value
    # smoothfilter: (gaussianfilter) filter for smoothing the noise pattern
    # losstype: type of loss to use ('psnr' or 'ssmi')
    # l1_lambda: weight for L1 regularization
    # num_iterations: number of iterations to run the optimization
    # model: the neural network model
    # device: (optional) the device to run the optimization on (e.g., 'cuda:0')
    # mask: (optional) binary mask to apply noise
    #
    # Anh-Huy Phan

    # If no device is provided, use the device of the input tensor 'x'
    if device is None:
        device = x.device

    # If no mask is provided, use a scalar value of 1 to apply noise uniformly
    if mask is None:
        mask = 1


    if learningrate is None:
        learningrate = 0.1
        
    # Initialize the noise pattern as a parameter
    if initial_noise is None:
        noise_pattern = torch.nn.Parameter(errbound * torch.randn_like(x) * mask).to(device)
    else:
        noise_pattern = torch.nn.Parameter(initial_noise * mask).to(device)

    # Apply the mask
    # noise_pattern = noise_pattern * mask

    # Define the optimizer
    optimizer = torch.optim.SGD([noise_pattern], lr=learningrate)

    # Define the maximum possible pixel value of the image
    MAX_I = 1.0

    # Calculate the number of pixels
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

    try:
        for iteration in range(num_iterations):
            optimizer.zero_grad()
    
            # Clamp the noise pattern values to ensure they stay within a valid range
            #noise_pattern.data.clamp_(-errbound, errbound)
            noise_pattern2 = errbound*torch.tanh(noise_pattern)
    
            # Smooth the noise pattern
            # kernel_size = smoothfilter.shape[-1]
            # smoothed_noise_pattern = F.conv2d(noise_pattern2 * mask, smoothfilter, padding=kernel_size // 2, groups=x.size(1))
            smoothed_noise_pattern = noise_pattern2 * mask
    
            # Apply current noise pattern
            perturbed_image = x + smoothed_noise_pattern
            
            # perturbed_image = torch.log(torch.exp(x) + smoothed_noise_pattern)
    
            # Forward pass through the model
            output = model(perturbed_image)
    
            # output['x_hat'] for the reconstructed image
            perturbed_output = output['x_hat']
    
            if losstype == 'psnr':
                # Calculate MSE loss
                mse_loss = F.mse_loss(perturbed_output, perturbed_image)
    
                # Calculate PSNR loss
                psnr_loss = 10 * torch.log10((MAX_I ** 2) / mse_loss)
    
                # maximize distortion = minimize PSNR
                loss = -psnr_loss  # Negative sign because we want to maximize PSNR
    
            elif losstype == 'ssim':
                # maximize distortion = minimize 1-SSIM
                ssim_perturbed = ssim(perturbed_output, x)
    
                loss = ssim_perturbed  # Negative sign because we want to maximize SSIM
    
            elif losstype == 'dists':
                
                dists_perturbed = dists(x, perturbed_output)    
                loss = dists_perturbed    
    
            if l1_lambda>0:
                # L1-norm for sparsity     
                l1norm = smoothed_noise_pattern.abs().sum()
                # Combine the loss with sparse L1 norm
                combined_loss = loss + l1_lambda * l1norm
            else:
                combined_loss = loss
    
            # Compute the bpp loss
            bpploss = bpp_loss(output, num_pixels)
    
            # Perform gradient descent
            combined_loss.backward()
            optimizer.step()
    
            # Print the loss every 100 iterations
            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss: {loss.item()}, BPP {bpploss}')

    except KeyboardInterrupt:
        # Save checkpoint on interruption
        save_checkpoint({'noise_pattern': noise_pattern.data, \
                         'smoothed_noise_pattern': smoothed_noise_pattern.data,\
                         'perturbed_image': perturbed_image.data,\
                         'perturbed_output': perturbed_output.data,\
                         'iteration': iteration})
        print("Interrupted, checkpoint saved.")
        return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern

 
