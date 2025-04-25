# MAE No-Propagation Diffusion Model

This project implements a Masked Autoencoder (MAE) with no-propagation diffusion for image reconstruction and generation on CIFAR-10 and other datasets.

## Overview

The implementation uses a Vision Transformer (ViT) architecture with masked autoencoding combined with denoising diffusion principles. The key difference from standard diffusion models is that this implementation uses a "no-propagation" approach where each denoising step is independent.

## Features

- Masked Vision Transformer (ViT) architecture
- EDM (Elucidating Diffusion Models) noise scheduling
- Independent denoising steps without error propagation
- Support for multiple datasets (CIFAR-10, Tiny-ImageNet, ImageNet-100)
- Visualization of reconstruction process
- Mixed precision training support
- Gradient checkpointing for memory efficiency

## Requirements

- PyTorch
- torchvision
- timm
- matplotlib
- seaborn
- einops
- numpy

## Usage

### Training

```bash
python train_mae_cifar10_noprop.py \
    --dataset cifar10 \
    --data_path /path/to/data \
    --batch_size 128 \
    --lr 1.5e-4 \
    --mask_ratio 0.75 \
    --T 4 \
    --epochs 1600 \
    --output_dir output/mae_cifar10_noprop
```

### Key Arguments

- `--dataset`: Dataset to use (cifar10, tiny-imagenet, imagenet-100)
- `--T`: Number of diffusion steps
- `--mask_ratio`: Ratio of masked patches
- `--vis_type`: Type of visualization (stochastic, heun)
- `--use_amp`: Enable mixed precision training
- `--use_checkpoint`: Enable gradient checkpointing
- `--resume`: Path to checkpoint for resuming training

## Model Architecture

The model consists of:
1. MAE Encoder: ViT-based encoder that extracts features from masked images
2. Multiple Denoising Decoders: Independent decoders for each denoising step
3. EDM Preconditioner: Scales model outputs according to EDM principles

## Visualization

The model generates visualizations of the reconstruction process at each epoch, showing:
- Original images
- Final reconstructed images 
- Intermediate steps of the diffusion process

## Citation

If you use this code, please cite the original MAE and EDM papers:

```
@article{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={CVPR},
  year={2022}
}

@article{karras2022elucidating,
  title={Elucidating the Design Space of Diffusion-Based Generative Models},
  author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  journal={NeurIPS},
  year={2022}
}
``` 