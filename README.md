# Adversarial Attack on Image Compression

This project implements adversarial attacks (FGSM and PGD) on AI-based image compression models to analyze their robustness and vulnerability to perturbations.

## Overview

The project demonstrates how adversarial examples can degrade the performance of neural image compression models, specifically targeting models from the CompressAI library. It includes implementations of:

- **FGSM (Fast Gradient Sign Method)**: Single-step adversarial attack
- **PGD (Projected Gradient Descent)**: Multi-step iterative adversarial attack
- **Multiple loss functions**: MSE, PSNR, SSIM, and BPP-based attacks
- **Comprehensive evaluation**: Metrics comparison and visualization

## Features

- Support for multiple AI compression models (Cheng2020-Anchor, etc.)
- Configurable attack parameters (epsilon, alpha, iterations)
- Multiple loss functions for generating adversarial examples
- Comprehensive metrics evaluation (MSE, PSNR, SSIM)
- Visualization of original vs adversarial images and their compressed versions
- Batch processing capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arofenitra/Adversarial_Attack_Image_Compression 
cd Adversarial_Attack_Image_Compression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run adversarial attacks on image compression models:

```bash
python3 scripts/lib.py --image_path path/to/your/image.png --save_path output_directory
```

#### Available Parameters

- `--image_path`: Path to input image (required)
- `--attack_type`: Type of attack (`fgsm`, `pgd`, or `both`) - default: `both`
- `--epsilon`: Perturbation budget (default: 8/255)
- `--alpha`: PGD step size (default: 8/255)
- `--num_iter`: Number of PGD iterations (default: 100)
- `--loss_type`: Loss function for attack (`mse`, `psnr`, `ssim`, `bpp`, or `all`) - default: `all`
- `--save_path`: Directory to save results (optional)
- `--model_name`: Compression model name (default: `cheng2020-anchor`)
- `--quality`: Model quality level (default: 6)

#### Example Commands

```bash
# Run both FGSM and PGD attacks with all loss types
python3 scripts/lib.py --image_path kodim/kodim01.png --save_path image_output

# Run only FGSM attack with MSE loss
python3 scripts/lib.py --image_path kodim/kodim01.png --attack_type fgsm --loss_type mse --save_path results

# Run PGD attack with custom parameters
python3 scripts/lib.py --image_path kodim/kodim01.png --attack_type pgd --epsilon 0.05 --alpha 0.01 --num_iter 50 --save_path results

# Test with different compression model and quality
python3 scripts/lib.py --image_path kodim/kodim01.png --model_name cheng2020-anchor --quality 8 --save_path results
```

### Jupyter Notebooks

The `notebooks/` directory contains:
- **fgsm_implementation.ipynb**: Interactive tutorials and visualizations
- Step-by-step implementation examples
- Detailed analysis of attack effectiveness
- Visualization tools for comparing results

## Project Structure

```
Adversarial_Attack_Image_Compression/
├── scripts/
│   └── lib.py                 # Main attack implementation
├── notebooks/
│   └── fgsm_implementation.ipynb  # Tutorials and visualizations
├── image_output/              # Generated results (if using --save_path)
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Attack Methods

### FGSM (Fast Gradient Sign Method)
- Single-step attack using the sign of gradients
- Fast but less effective than iterative methods
- Good for quick evaluation of model robustness

### PGD (Projected Gradient Descent)
- Multi-step iterative attack
- More effective but computationally expensive
- Projects perturbations back to valid range at each step

### Loss Functions
- **MSE**: Mean Squared Error between original and compressed images
- **PSNR**: Peak Signal-to-Noise Ratio (maximized by minimizing MSE)
- **SSIM**: Structural Similarity Index (perceptual quality metric)
- **BPP**: Bits Per Pixel (compression efficiency metric)

## Output

The script generates:
1. **Console output**: Attack parameters, metrics comparison, and performance statistics
2. **Visualizations**: 2x2 grid showing original, adversarial, and their compressed versions
3. **Saved images**: Compressed adversarial images (if `--save_path` specified)

### Metrics Reported
- Reconstruction MSE before and after attack
- PSNR values
- MSE increase ratio
- SSIM and other perceptual metrics
- Perturbation statistics (max, average)

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- `torch`: PyTorch framework
- `compressai`: AI image compression models
- `torchvision`: Computer vision utilities
- `numpy<2`: Numerical computing
- `PIL`: Image processing
- `matplotlib`: Visualization
- `kornia`: Computer vision library

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

Free of use


## Acknowledgments

- CompressAI library for providing pre-trained compression models
- PyTorch team for the deep learning framework
- Original FGSM and PGD paper authors
