import os
import sys
import argparse

from PIL import Image 
import torch
from torchvision import transforms
import pytorch_wavelets
import torch.nn as nn
import numpy as np
from skimage import io
from skimage.util import img_as_float

from compressai.zoo import models
from kornia.losses import SSIMLoss, PSNRLoss

from module_load import *

# Clear GPU memory
torch.cuda.empty_cache()

MAX_I = 1.0

class MultiScaleDecomposition:
    def __init__(self, wavelet='haar', device='cpu', scales = 1):
        self.dwt = pytorch_wavelets.DWTForward(J=scales, wave=wavelet).to(device)
        self.device = device

    def decompose(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        # Decompose x into low and high-frequency components
        low, details = self.dwt(x)
        return low, details

class Reconstruction:
    def __init__(self, wavelet='haar', device='cpu'):
        self.idwt = pytorch_wavelets.DWTInverse(wave=wavelet).to(device)
        self.device = device

    def reconstruct(self, low, perturbed_details):
        # Ensure inputs are on the correct device
        low = low.to(self.device)
        perturbed_details = [d.to(self.device) for d in perturbed_details]
        # Reconstruct image
        return self.idwt((low, perturbed_details))

def reduce_noise_level(
    x, model, device, psnr, desired_noiselevel=0.02, current_noiselevel=0.05, g = "tanhatanh
):
    # Initialize variables
    psnr_ = [psnr(torch.clamp(perturbed_output,0,1), perturbed_image).detach().cpu().numpy()]
    attack_area_ = []

    qualitymeasure = 'psnr'
    num_iterations = 1000
    l1_lambda = 0.000  # Increase l1_weight to reduce noise

    sigmas = np.linspace(current_noiselevel, desired_noiselevel, 4)

    # Variables to keep track of the last positive psnr_k and its corresponding parameters
    last_positive_psnr_k = None
    last_positive_attack_area = None
    last_positive_params = None

    for current_noiselevel_ in sigmas:
        print(current_noiselevel_)

        for krun in range(5):
            noise_pattern_bk = noise_pattern.clone()
    
            # Learn new noise mask
            mask, mask_noise = noise_to_mask(smoothed_noise_pattern.clone())
    
            # Smooth the mask
            mask = F.conv2d(mask, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2] // 2, groups=x.shape[1])
            mask /= mask.max()  # Normalize the smoothed mask to have values between 0 and 1
    
            # Plot the mask
            # Plot the mask
            #plt.figure(krun)
            #plt.imshow(mask.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
            #plt.axis('off')
            #plt.show()
            #plt.pause(0.1)
    
            new_attack_area = sum(mask_noise.ravel())
            print(f'Attack area {new_attack_area}')
            attack_area_.append(new_attack_area)

        if new_attack_area == 0:
            break
        (
            perturbed_image,
            perturbed_output,
            smoothed_noise_pattern,
            noise_pattern,
        ) = new_maxdistortion(
            x,
            errbound=current_noiselevel_,
            smoothfilter=None,
            losstype=qualitymeasure,
            l1_lambda=l1_lambda,
            num_iterations=num_iterations,
            model=model,
            device=device,
            mask=mask,
            initial_noise=noise_pattern,
            learningrate=0.1,
            g = "tanhatanh"
        )

            psnr_k = psnr(perturbed_output, perturbed_image).detach().cpu().numpy()

            print(f"{psnr_k}")
            psnr_.append(psnr(torch.clamp(perturbed_output,0,1), perturbed_image).detach().cpu().numpy())

            if psnr_k > 0:
                last_positive_psnr_k = psnr_k
                # last_positive_attack_area = new_attack_area
                last_positive_params = {
                    "perturbed_image": perturbed_image.clone(),
                    "perturbed_output": perturbed_output.clone(),
                    'smoothed_noise_pattern': smoothed_noise_pattern.clone(),
                    'noise_pattern': noise_pattern.clone(),
                    'mask': mask.clone()
                    "current_noiselevel": current_noiselevel_,
                }

            if psnr_k < 50:
                noise_pattern = noise_pattern_bk
                current_noiselevel_ = last_positive_params["current_noiselevel"]
                (
                    perturbed_image,
                    perturbed_output,
                    smoothed_noise_pattern,
                    noise_pattern,
                ) = new_maxdistortion(
                    x,
                    errbound=current_noiselevel_,
                    smoothfilter=None,
                    losstype=qualitymeasure,
                    l1_lambda=l1_lambda,
                    num_iterations=num_iterations,
                    model=model,
                    device=device,
                    mask=mask,
                    initial_noise=noise_pattern,
                    learningrate=0.1,
                    g = "tanhatanh"
                )

                break

            if psnr_k < 0:
                break

        if psnr_k < 0:
            break

    # Save the results to the original variables
    if last_positive_params:
        perturbed_image = last_positive_params["perturbed_image"]
        perturbed_output = last_positive_params["perturbed_output"]
        smoothed_noise_pattern = last_positive_params["smoothed_noise_pattern"]
        noise_pattern = last_positive_params["noise_pattern"]
        mask = last_positive_params["mask"]

    return (
        perturbed_image,
        perturbed_output,
        smoothed_noise_pattern,
        noise_pattern,
        #last_positive_params,
        noise_pattern_bk,
        mask,
        psnr_
    )
#spatial attack
def maxdistortion_step(
    x, model, device, psnr, current_noiselevel, qualitymeasure, 
    l1_lambda, num_iterations, scales, initial_noise=None, mask=mask
):
    perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
        new_maxdistortion(
            x,
            errbound=current_noiselevel,
            smoothfilter=None,
            losstype=qualitymeasure,
            l1_lambda=l1_lambda,
            num_iterations=num_iterations,
            model=model,
            device=device,
            mask=mask,
            initial_noise=None,
            learningrate=0.1,
            g = "tanharctanh" 
        )

    psnr_1 = psnr(perturbed_output, perturbed_image).detach().cpu().numpy()
    while psnr_1 < 100:
        perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
            new_maxdistortion(
                x,
                errbound=current_noiselevel,
                smoothfilter=None,
                losstype=qualitymeasure,
                l1_lambda=0.0,
                num_iterations=num_iterations,
                model=model,
                device=device,
                mask=mask,
                initial_noise=noise_pattern,
                learningrate=0.1,
                g = "tanharctanh"
            )

        psnr_1 = psnr(perturbed_output, perturbed_image).detach().cpu().numpy()

    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern, mask
    
def shrink_step(x, device, perturbed_image, perturbed_output, scales, wavelet='haar'):
    # Define the size and standard deviation of the Gaussian kernel
    gaussian_filter_mask = create_gaussian_filter_mask(x,device)
    residue = perturbed_output-perturbed_image
    new_mask = mask_from_residue(residue)
    psnr_1 = psnr(perturbed_output,perturbed_image).detach().cpu().numpy()
    while psnr_1<100:
        perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
            new_maxdistortion(
                x,
                errbound=current_noiselevel,
                smoothfilter=None,
                losstype=qualitymeasure,
                l1_lambda=0.0,
                num_iterations=num_iterations,
                model=model,
                device=device,
                mask=new_mask,
                initial_noise=noise_pattern,
                learningrate=0.1,
                g = "tanharctanh"
            )

        psnr_1 = psnr(perturbed_output, perturbed_image).detach().cpu().numpy()
    for krun in range(100):
        noise_pattern_bk = noise_pattern.clone()
        erroded_mask, eroded_mask_2d = create_erroded_mask(smoothed_noise_pattern)
    
        # plt.imshow(mask_noise)
        #plt.figure(krun)
        #plt.imshow(mask.squeeze().cpu().detach().numpy().transpose(1, 2, 0));
        #plt.axis('off')
        #plt.show
        #plt.pause(0.1)
        
        new_attack_area = sum(eroded_mask_2d.ravel())
        print(f'Attack area {new_attack_area}')
        attack_area_.append(new_attack_area)
    
        if new_attack_area <= 10000:
            break    
    
       # perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
          #  maxdistortion_multiplicativenoise(x, sigma, gaussian_filter, qualitymeasure, l1_lambda, num_iterations, \
              #            model, initial_noise=noise_pattern.clone(), mask= eroded_mask)
        perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
            new_maxdistortion(x, errbound=current_noiselevel, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=num_iterations, \
                                          model=model, device=device, mask=eroded_mask,initial_noise=noise_pattern,learningrate=0.1,g="tanhatanh")
        psnr_k = psnr(perturbed_output,perturbed_image).detach().cpu().numpy()
        
    
        print(f'{psnr_k}')
        psnr_.append(psnr_k)    
    
        if psnr_k<0:   #why not eroded mask?????
            noise_pattern = noise_pattern_bk
            #perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
            #maxdistortion_multiplicativenoise(x, sigma, gaussian_filter, qualitymeasure, l1_lambda, num_iterations, \
                        #  model, initial_noise=noise_pattern.clone(), mask= mask)
            perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
            new_maxdistortion(x, errbound=current_noiselevel, smoothfilter = None, losstype = 'psnr', l1_lambda=0, num_iterations=num_iterations, \
                                          model=model, device=device, mask=eroded_mask,initial_noise=noise_pattern,learningrate=0.1,g="tanhatanh")
            break
    
    return perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern, new_mask
    
    


def shrink_step_multiscale(x, device, perturbed_image, perturbed_output, scales, wavelet='haar'):
    # Define the size and standard deviation of the Gaussian kernel
    kernel_size = 5
    sigma_filter = 1.0

    # Create the Gaussian kernel
    gaussian_filter = gaussian_kernel(kernel_size, sigma_filter)

    # Add batch and channel dimensions to the filter
    gaussian_filter = gaussian_filter.view(1, 1, *gaussian_filter.size())

    # Assuming 'image' is with shape [batch_size, channels, height, width]
    # Repeat the filter for each input channel
    gaussian_filter = gaussian_filter.repeat(x.size(1), 1, 1, 1)
    gaussian_filter = gaussian_filter.to(device)
    

    # Gaussian blur to the mask to smooth the edges
    kernel_size = 21
    sigma_filter = 8

    # Create the Gaussian kernel
    gaussian_filter_mask = gaussian_kernel(kernel_size, sigma_filter)

    # Add batch and channel dimensions to the filter
    gaussian_filter_mask = gaussian_filter_mask.view(1, 1, *gaussian_filter_mask.size())

    # Assuming 'image' is with shape [batch_size, channels, height, width]
    # Repeat the filter for each input channel
    gaussian_filter_mask = gaussian_filter_mask.repeat(x.size(1), 1, 1, 1)
    gaussian_filter_mask = gaussian_filter_mask.to(device)

    residue = perturbed_output-perturbed_image

    residue = F.conv2d(residue, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2]//2,groups=x.shape[1])
    residue = F.conv2d(residue, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2]//2,groups=x.shape[1])

    ell2_noise = torch.sqrt(torch.sum(residue**2,dim = 1));

    mask_noise = ell2_noise>.5#torch.max(ell2_noise) * 1e-1
    mask_noise = mask_noise.squeeze()

    mask_noise = mask_noise.cpu().detach().numpy()
    mask_noise = opening(mask_noise, square(10))
    # mask_noise = opening(mask_noise, square(10))
    mask_noise = dilation(mask_noise, square(20))
    # mask_noise = dilation(mask_noise, square(5))

    # plt.imshow(mask_noise)

    # Mask 3D 
    new_mask = torch.zeros_like(perturbed_image)
    nnz_ix = np.where(mask_noise==1)
    new_mask[:,:,nnz_ix[0],nnz_ix[1]] = 1

    # new_mask,new_2dmask = noise_to_mask(residue)
    new_mask = F.conv2d(new_mask, gaussian_filter_mask, padding=gaussian_filter_mask.shape[2]//2,groups=x.shape[1])

    mask_scale = [None] * (scales+1)
    for kscale  in range(scales):
        decomposer = MultiScaleDecomposition(wavelet=wavelet, device=device,scales = kscale+1)
        reconstructor = Reconstruction(wavelet=wavelet, device=device)
        # Decompose input image
        
        low_mask_kscale, details_mask = decomposer.decompose(new_mask)
        low_mask_kscale = low_mask_kscale.abs()> .1
        # Convert to binary tensor of 0s and 1s
        low_mask_kscale = low_mask_kscale.to(torch.float)  # Use torch.int if integers are needed

        if kscale == scales-1:
            mask_scale[0] = low_mask_kscale
            
        mask_scale[kscale+1] = low_mask_kscale.unsqueeze(1).expand(-1,  3, -1,-1,-1)  

    return mask_scale

def defense(perturbed_image, model, device, psnr):
    perturbed_image_nograd = perturbed_image.detach()
    sigmas = np.linspace(0.2,0.01,10)
    qualitymeasure = 'psnr'
    l1_lambda = 0.0
    num_iterations = 2000
    quality_loss_lambda = 0.1

    with torch.no_grad():
        modeloutput = model(perturbed_image_nograd)

    decompress_perturbed_image = modeloutput['x_hat']

    psnr_ = []
    bpp_f_ = []

    for sigma in sigmas:

        perturbed_image_def, perturbed_output_def, smoothed_noise_pattern_def, noise_pattern_def= \
            new_defend_maxdistortion(perturbed_image_nograd, sigma, None, 'psnr', \
            l1_lambda, num_iterations, model, device=device, mask=None,initial_noise=None,learningrate = 0.1,g="tanhatanh")
        
        psnr_k = psnr(perturbed_output_def.clamp(0,1),perturbed_image_def.clamp(0,1))
        psnr_.append(psnr_k.detach().cpu().numpy())
        

    ix = np.argmin(psnr_)
    sigma = sigmas[ix]
    # sigma = 0.02
    perturbed_image_def, perturbed_output_def, smoothed_noise_pattern_def, noise_pattern_def= \
            new_defend_maxdistortion(perturbed_image_nograd, sigma, None, 'psnr', \
            l1_lambda, num_iterations, model, device=device, mask=None,initial_noise=None,learningrate = 0.1,g="tanhatanh")
        
    psnr_k = psnr(perturbed_output_def,decompress_perturbed_image)

    return perturbed_image_def, perturbed_output_def, smoothed_noise_pattern_def, noise_pattern_def, psnr_k

def quantization(noise_pattern, x, model, sigma):
    noise_level = sigma
    quantized_smoothed_noise_pattern = noise_pattern.clone()
    quantized_smoothed_noise_pattern[noise_pattern<-0.01] = -noise_level
    quantized_smoothed_noise_pattern[noise_pattern>0.01] = noise_level
    quantized_smoothed_noise_pattern[(noise_pattern<=0.01)*(noise_pattern>=-0.01)] = 0

    quantized_perturbed_image = x + quantized_smoothed_noise_pattern

    with torch.no_grad():
        quantizedattack_output = model(quantized_perturbed_image)

    quantized_perturbed_output = quantizedattack_output['x_hat']

    return  quantized_perturbed_image, quantized_perturbed_output,quantized_smoothed_noise_pattern, noise_pattern

def run_experiment(args):
    image_path = args.image_path
    model_name = "cheng2020-anchor" #'cheng2020-attn' 'cheng2020-anchor' 'mbt2018' 'mbt2018-mean'  'bmshj2018-hyperprior'  'bmshj2018_factorized_relu' 'bmshj2018_factorized'
    model_name2 = model_name.replace('-', '_')
    quality = 6

    ############### load the model ###############
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Clear GPU memory
    torch.cuda.empty_cache()
    model_class = models[model_name]

    # Set compression-decompression quality for AI image compression 
    model = model_class(quality=quality, pretrained=True).to(device)
    for param in model.parameters():
        param.requires_grad = False

    ############### load Losses ###############
    psnr = PSNRLoss(max_val=1.0)

    ############### load the image ###############
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([transforms.ToTensor()])
    x = preprocess(img).unsqueeze(0).to(device)
    img_rgb = img_as_float(io.imread(image_path)) 

    ############### First step ###################
    qualitymeasure = 'psnr'
    l1_lambda = 0.00
    num_iterations = 2000
    current_noiselevel = 0.05

    scales = 3

    perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = maxdistortion_step(
        x, model, device, psnr, current_noiselevel, qualitymeasure, l1_lambda, num_iterations, scales
    )
    ############### export results ###############
    methodname = "%s_q%d_sigma%.2f_minpsnr_masksmooth_logexpnoise_multiscale%d" % (model_name2, quality, current_noiselevel, scales)
    file_no_extension = os.path.splitext(image_path)[0]
    file_name = file_no_extension + methodname + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump((smoothed_noise_pattern, noise_pattern, [], perturbed_image,perturbed_output, []), f)

    ############### Second step: reduce noise level ###############
    desired_noiselevel = 0.02
    current_noiselevel = 0.05
    scales = 3
    perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern, last_positive_params, noise_pattern_bk, psnr_ = \
        reduce_noise_level(
            x, model, device, psnr, desired_noiselevel, current_noiselevel, scales
        )

    noise_pattern = [param.clone() for param in noise_pattern_bk]
    current_noiselevel = last_positive_params["current_noiselevel"]
    
    perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = maxdistortion_step(
        x, model, device, psnr, current_noiselevel, qualitymeasure, l1_lambda, num_iterations, scales, 
        initial_noise=noise_pattern
    )

    ############### export results ###############
    methodname = "%s_q%d_sigma%.2f_minpsnr_masksmooth_logexpnoise_multiscale%d" % (model_name2, quality, current_noiselevel, scales)
    file_no_extension = os.path.splitext(image_path)[0]
    file_name = file_no_extension + methodname + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump((smoothed_noise_pattern, noise_pattern, [], perturbed_image,perturbed_output, psnr_), f)

    ############### Third step: make shrink ###############
    mask_scale = shrink_step(x, device, perturbed_image, perturbed_output, scales)
    perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = maxdistortion_step(
        x, model, device, psnr, current_noiselevel, qualitymeasure, l1_lambda, num_iterations, scales, 
        initial_noise=noise_pattern, mask=mask_scale
    )

    ############### export results ###############
    methodname = "%s_q%d_sigma%.2f_minpsnr_masksmooth_logexpnoise_multiscale%d" % (model_name2, quality, current_noiselevel, scales)
    file_no_extension = os.path.splitext(image_path)[0]
    file_name = file_no_extension + methodname + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump((smoothed_noise_pattern, noise_pattern, mask_scale,perturbed_image,perturbed_output, psnr_), f)

    ############### Fourth step: quantization ###############
    quantized_smoothed_noise_pattern, _, _ = \
        quantization(noise_pattern, x, model, current_noiselevel)
    
    perturbed_image, perturbed_output, smoothed_noise_pattern, noise_pattern = \
        maxdistortion_step(
        x, model, device, psnr, current_noiselevel, qualitymeasure, l1_lambda, num_iterations, scales, 
        initial_noise=quantized_smoothed_noise_pattern, mask=mask_scale
    )

    ############### export results ###############
    methodname = "%s_q%d_sigma%.2f_minpsnr_masksmooth_logexpnoise_multiscale%d" % (model_name2, quality, current_noiselevel, scales)
    file_no_extension = os.path.splitext(image_path)[0]
    file_name = file_no_extension + methodname + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump((smoothed_noise_pattern, noise_pattern, mask_scale,perturbed_image,perturbed_output, psnr_), f)

    ############### Defense ###############
    perturbed_image_def, perturbed_output_def, smoothed_noise_pattern_def, noise_pattern_def, psnr_k = \
        defense(perturbed_image, model, device, psnr)

    ############### export results ###############
    methodname = "%s_q%d_sigma%.2f_minpsnr_masksmooth_logexpnoise_multiscale%d" % (model_name2, quality, current_noiselevel, scales)
    file_no_extension = os.path.splitext(image_path)[0]
    file_name = file_no_extension + methodname + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump((smoothed_noise_pattern_def, noise_pattern_def, mask_scale,perturbed_image_def,perturbed_output_def, psnr_k), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args)
