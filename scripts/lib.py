
import argparse
# Environment setup and imports
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import PIL 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn 
def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial Attacks on Compression Models')
    
    parser.add_argument('--image_path', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--attack_type', type=str, default='both',
                       choices=['fgsm', 'pgd', 'both'],
                       help='Attack type')
    parser.add_argument('--epsilon', type=float, default=8/255,
                       help='Perturbation budget')
    parser.add_argument('--alpha', type=float, default=8/255,
                       help='PGD step')
    parser.add_argument('--num_iter', type=int, default=100,
                       help='PGD iterations')
    parser.add_argument('--loss_type', type=str, default='all',
                       choices=['mse', 'psnr', 'ssim', 'bpp', "all"],
                       help='Loss function')
    parser.add_argument('--save_path', type=str, default=None, help='saving Perturbed output and perturbed input')
    parser.add_argument('--model_name', type=str, default='cheng2020-anchor',
                       help='Name of the AI compression model')
    parser.add_argument('--quality', type=int, default=6,
                       help='Quality of the AI model')   
    
    return parser.parse_args()
args = parse_args()


# Computational device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path):
    """Load image and convert to tensor"""
    image = PIL.Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image_tensor, image

def save_image(image_path,x):
    # x is a numpy image between [0,1]
    x = np.asarray(x)
    if x.dtype !=np.uint8:
        x = (np.clip(x,0,1)*255).round().astype(np.uint8)
    os.makedirs(os.path.dirname(image_path), exist_ok=True) if os.path.dirname(image_path) else None
    PIL.Image.fromarray(x).save(image_path)

def _eps_scalar():
    return torch.tensor(1e-12, device=next(model.parameters()).device if 'model' in globals() else 'cuda' if torch.cuda.is_available() else 'cpu')

class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val
        self.mse = nn.MSELoss()
    def forward(self, x, y):
        mse = self.mse(x, y)
        eps = _eps_scalar()
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse + eps))
        return -psnr

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    def _create_window(self, window_size, channel):
        def _gaussian(window_size, sigma):
            gauss = torch.tensor([math.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()
        _1D = _gaussian(window_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D.expand(channel, 1, window_size, window_size).contiguous()
    def forward(self, img1, img2):
        (_, ch, _, _) = img1.size()
        if ch == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, ch).to(img1.device).type_as(img1)
            self.window = window
            self.channel = ch
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=ch)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=ch)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=ch) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=ch) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=ch) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - (ssim_map.mean() if self.size_average else ssim_map.mean(1).mean(1).mean(1))

class BppLoss(nn.Module):
    def forward(self, output_dict, x):
        if 'bpp' in output_dict:
            return -output_dict['bpp']
        if 'likelihoods' in output_dict:
            total_bits = 0.0
            for lik in output_dict['likelihoods'].values():
                total_bits = total_bits + torch.sum(-torch.log2(lik.clamp_min(1e-9)))
            bpp = total_bits / (x.shape[0] * x.shape[2] * x.shape[3])
            return -bpp
        return -nn.MSELoss()(output_dict['x_hat'], x)

def get_loss_function(loss_type):
    if loss_type == 'mse':
        return nn.MSELoss()
    if loss_type == 'psnr':
        return PSNRLoss()
    if loss_type == 'ssim':
        return SSIMLoss()
    if loss_type == 'bpp':
        return BppLoss()
    raise ValueError(f'Unknown loss type: {loss_type}')

@torch.enable_grad()
def differentiable_forward(m, x):
    was_training = m.training
    m.train()
    out = m(x)
    if not was_training:
        m.eval()
    return out

@torch.enable_grad()
def fgsm_attack(model, x, epsilon, loss_type='mse', targeted=False, target_output=None):
    x_adv = x.clone().detach().requires_grad_(True)
    criterion = get_loss_function(loss_type)
    out = differentiable_forward(model, x_adv)
    if loss_type == 'bpp':
        loss = criterion(out, x_adv)
    else:
        target = target_output if (targeted and target_output is not None) else x
        loss = criterion(out['x_hat'], target)
    grad_multiplier = -1.0 if targeted else 1.0
    model.zero_grad(set_to_none=True)
    if x_adv.grad is not None:
        x_adv.grad.zero_()
    (grad_multiplier * loss).backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv

@torch.enable_grad()
def pgd_attack(model, x, epsilon, alpha, num_iter, loss_type='mse', targeted=False, target_output=None, random_start=True):
    if random_start:
        x_adv = torch.clamp(x + torch.empty_like(x).uniform_(-epsilon, epsilon), 0.0, 1.0)
    else:
        x_adv = x.clone()
    criterion = get_loss_function(loss_type)
    for _ in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        out = differentiable_forward(model, x_adv)
        if loss_type == 'bpp':
            loss = criterion(out, x_adv)
        else:
            target = target_output if (targeted and target_output is not None) else x
            loss = criterion(out['x_hat'], target)
        grad_multiplier = -1.0 if targeted else 1.0
        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        (grad_multiplier * loss).backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
    return x_adv.detach()


# Visualization function
def visualize_attack(original, adversarial, original_compressed, adversarial_compressed, epsilon,loss_type="psnr"):
    """Visualize original and adversarial images with their compressed versions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert tensors to numpy for plotting
    original_np = original.squeeze(0).cpu().permute(1, 2, 0).numpy()
    adversarial_np = adversarial.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Plot original and adversarial
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(adversarial_np)
    axes[0, 1].set_title(f'Adversarial Image (ε={epsilon:.02f}),loss_type={loss_type}')
    axes[0, 1].axis('off')
    
    # Plot compressed versions
    if 'x_hat' in original_compressed:
        orig_comp_np = original_compressed['x_hat'].squeeze(0).cpu().permute(1, 2, 0).numpy()
        adv_comp_np = adversarial_compressed['x_hat'].squeeze(0).cpu().permute(1, 2, 0).numpy()
        
        axes[1, 0].imshow(np.clip(orig_comp_np, 0, 1))
        axes[1, 0].set_title('Compressed Original')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.clip(adv_comp_np, 0, 1))
        axes[1, 1].set_title('Compressed Adversarial')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print metrics
    perturbation = torch.abs(adversarial - original)
    max_perturbation = perturbation.max().item()
    avg_perturbation = perturbation.mean().item()
    
    print(f"Attack parameters: ε={epsilon}")
    print(f"Maximum perturbation: {max_perturbation:.4f}")
    print(f"Average perturbation: {avg_perturbation:.4f}")

def get_metrics(x,y):
    x,y = x.detach(),y.detach()
    mse_loss = nn.MSELoss()(x,y).item()
    psnr_loss = -PSNRLoss()(x,y).item()
    ssim_loss = SSIMLoss()(x,y).item()
    return {"mse":mse_loss,
        "psnr":psnr_loss,
        # "bpp": bpp_loss,
        "ssim":1-ssim_loss    
        }

print(f"Current path: {os.getcwd()}\nInside")
try:
    os.chdir(os.path.dirname(os.getcwd()))
except Exception:
    pass
# compressai
try:
    from compressai.zoo import cheng2020_anchor
    ## Load pretrained AI image compression models
    # from compressai.zoo import cheng2020_anchor
    from compressai.zoo import models
except Exception as e:
    raise SystemExit("compressai is required. Install via: pip install compressai[full]")

# device



import warnings; warnings.filterwarnings("ignore")
model_name = args.model_name

# Dynamically retrieve the model class
quality = args.quality
model_class = models[model_name]

# Clear GPU memory
torch.cuda.empty_cache()
# Set compression-decompression quality for AI image compression 
model = model_class(quality=quality, pretrained=True).to(device)
for param in model.parameters():
    param.requires_grad = False

# Loading image
transform = transforms.Compose([
    transforms.ToTensor(),
])


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load image
image_path = args.image_path
file_name =  os.path.splitext(os.path.basename(image_path))[0]
x_original, pil_image = load_image(image_path)

# Compress the images to see the effect
print("Compressing images...")
with torch.no_grad():
    # Original compression
    comp_original = model.compress(x_original)
    decomp_original = model.decompress(comp_original['strings'], comp_original['shape'])


# Define loss function
criterion = nn.MSELoss()

# Attack parameters
epsilon = args.epsilon
alpha = args.alpha
num_iter = args.num_iter

print("Performing FGSM attack...")
# FGSM Attack
if args.loss_type=="all":
    loss_types = ["mse","psnr","ssim","bpp"]
elif args.loss_type == "mse":
    loss_types = ["mse"]
elif args.loss_type == "psnr":
    loss_types = ["psnr"]
elif args.loss_type == "ssim":
    loss_types = ["ssim"]
elif args.loss_type == "bpp":
    loss_types = ["bpp"]
else:
    raise ValueError("Incorrect loss type")
for loss_type in loss_types:
    print(f"\n\nLoss_type: {loss_type}")
    if args.attack_type == "both":
        x_adv_fgsm = fgsm_attack(model, x_original, epsilon, loss_type=loss_type, targeted=False, target_output=None)
        decomp_fgsm = model(x_adv_fgsm)


        # print("PGD Attack Results:")
        # visualize_attack(x_original, x_adv_pgd, decomp_original, decomp_pgd, epsilon)

        # Calculate quantitative metrics
        mse_original = criterion(decomp_original['x_hat'], x_original).item()
        mse_fgsm = criterion(decomp_fgsm['x_hat'], x_original).item()
        # mse_pgd = criterion(decomp_pgd['x_hat'], x_original).item()

        print(f"\nReconstruction MSE:")
        print(f"Original: {mse_original:.6f}")
        print(f"After FGSM: {mse_fgsm:.6f} with psnr(oi,ao)={-10*np.log10(mse_fgsm):.2f}")
        # print(f"After PGD: {mse_pgd:.6f} with psnr(oi,ao)={-10*np.log10(mse_pgd):.2f}")
        print(f"FGSM Increase: {mse_fgsm/mse_original:.2f}x")
        # print(f"PGD Increase: {mse_pgd/mse_original:.2f}x")
        print(f"metrics between x and clamp(f(x_adv)): \n{get_metrics(x_original, decomp_fgsm['x_hat'].clamp(0, 1))}")
        visualize_attack(x_original, x_adv_fgsm, decomp_original, decomp_fgsm, epsilon,loss_type=loss_type)



        print("Performing PGD attack...")
        # PGD Attack
        print(f"\n\nLoss_type: {loss_type}")
        x_adv_pgd = pgd_attack(model, x_original, epsilon, alpha=alpha, num_iter=num_iter, loss_type=loss_type, targeted=False, target_output=None)
        decomp_pgd = model(x_adv_pgd)


        # print("PGD Attack Results:")
        # visualize_attack(x_original, x_adv_pgd, decomp_original, decomp_pgd, epsilon)

        # Calculate quantitative metrics
        mse_original = criterion(decomp_original['x_hat'], x_original).item()
        mse_pgd = criterion(decomp_pgd['x_hat'], x_original).item()
        # mse_pgd = criterion(decomp_pgd['x_hat'], x_original).item()

        print(f"\nReconstruction MSE:")
        print(f"Original: {mse_original:.6f}")
        print(f"After PGD: {mse_pgd:.6f} with psnr(oi,ao)={-10*np.log10(mse_pgd):.2f}")
        # print(f"After PGD: {mse_pgd:.6f} with psnr(oi,ao)={-10*np.log10(mse_pgd):.2f}")
        print(f"PGD Increase: {mse_pgd/mse_original:.2f}x")
        # print(f"PGD Increase: {mse_pgd/mse_original:.2f}x")
        print(f"metrics between x and clamp(f(x_adv)): \n{get_metrics(x_original, decomp_pgd['x_hat'].clamp(0, 1))}")
        visualize_attack(x_original, x_adv_pgd, decomp_original, decomp_pgd, epsilon,loss_type=loss_type)
        if args.save_path is not None and os.path.exists(args.save_path):
            save_image(f"{args.save_path}/decomp_fgsm_{args.model_name}_q{args.quality}_eps{args.epsilon}_{loss_type}_{file_name}.png", decomp_fgsm['x_hat'].clamp(0, 1).detach().squeeze(0).cpu().permute(1, 2, 0).numpy())
        else:
            pass
        if args.save_path is not None and os.path.exists(args.save_path):
            save_image(f"{args.save_path}/decomp_pgd_{args.model_name}_q{args.quality}_eps{args.epsilon}_alpha{args.alpha}_{loss_type}_{file_name}.png", decomp_pgd['x_hat'].clamp(0, 1).detach().squeeze(0).cpu().permute(1, 2, 0).numpy())
        else:
            pass
    elif args.attack_type =="fgsm":
        x_adv_fgsm = fgsm_attack(model, x_original, epsilon, loss_type=loss_type, targeted=False, target_output=None)
        decomp_fgsm = model(x_adv_fgsm)


        # print("PGD Attack Results:")
        # visualize_attack(x_original, x_adv_pgd, decomp_original, decomp_pgd, epsilon)

        # Calculate quantitative metrics
        mse_original = criterion(decomp_original['x_hat'], x_original).item()
        mse_fgsm = criterion(decomp_fgsm['x_hat'], x_original).item()
        # mse_pgd = criterion(decomp_pgd['x_hat'], x_original).item()

        print(f"\nReconstruction MSE:")
        print(f"Original: {mse_original:.6f}")
        print(f"After FGSM: {mse_fgsm:.6f} with psnr(oi,ao)={-10*np.log10(mse_fgsm):.2f}")
        # print(f"After PGD: {mse_pgd:.6f} with psnr(oi,ao)={-10*np.log10(mse_pgd):.2f}")
        print(f"FGSM Increase: {mse_fgsm/mse_original:.2f}x")
        # print(f"PGD Increase: {mse_pgd/mse_original:.2f}x")
        print(f"metrics between x and clamp(f(x_adv)): \n{get_metrics(x_original, decomp_fgsm['x_hat'].clamp(0, 1))}")
        visualize_attack(x_original, x_adv_fgsm, decomp_original, decomp_fgsm, epsilon,loss_type=loss_type)
        if args.save_path is not None and os.path.exists(args.save_path):
            save_image(f"{args.save_path}/decomp_fgsm_{args.model_name}_q{args.quality}_eps{args.epsilon}_{loss_type}_{file_name}.png", decomp_fgsm['x_hat'].clamp(0, 1).detach().squeeze(0).cpu().permute(1, 2, 0).numpy())
        else:
            pass
    elif args.attack_type=="pgd":
        
        print("Performing PGD attack...")
        # PGD Attack
        print(f"\n\nLoss_type: {loss_type}")
        x_adv_pgd = pgd_attack(model, x_original, epsilon, alpha=alpha, num_iter=num_iter, loss_type=loss_type, targeted=False, target_output=None)
        decomp_pgd = model(x_adv_pgd)


        # print("PGD Attack Results:")
        # visualize_attack(x_original, x_adv_pgd, decomp_original, decomp_pgd, epsilon)

        # Calculate quantitative metrics
        mse_original = criterion(decomp_original['x_hat'], x_original).item()
        mse_pgd = criterion(decomp_pgd['x_hat'], x_original).item()
        # mse_pgd = criterion(decomp_pgd['x_hat'], x_original).item()

        print(f"\nReconstruction MSE:")
        print(f"Original: {mse_original:.6f}")
        print(f"After PGD: {mse_pgd:.6f} with psnr(oi,ao)={-10*np.log10(mse_pgd):.2f}")
        # print(f"After PGD: {mse_pgd:.6f} with psnr(oi,ao)={-10*np.log10(mse_pgd):.2f}")
        print(f"PGD Increase: {mse_pgd/mse_original:.2f}x")
        # print(f"PGD Increase: {mse_pgd/mse_original:.2f}x")
        print(f"metrics between x and clamp(f(x_adv)): \n{get_metrics(x_original, decomp_pgd['x_hat'].clamp(0, 1))}")
        visualize_attack(x_original, x_adv_pgd, decomp_original, decomp_pgd, epsilon,loss_type=loss_type)
        if args.save_path is not None and os.path.exists(args.save_path):
            save_image(f"{args.save_path}/decomp_pgd_{args.model_name}_q{args.quality}_eps{args.epsilon}_alpha{args.alpha}_{loss_type}_{file_name}.png", decomp_pgd['x_hat'].clamp(0, 1).detach().squeeze(0).cpu().permute(1, 2, 0).numpy())
        else:
            pass
