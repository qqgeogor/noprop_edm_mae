# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

import os
import sys
import time
import math
import json
import random
import datetime
import subprocess
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from collections import defaultdict, deque
from pathlib import Path
from torch import nn
from PIL import ImageFilter, ImageOps, Image, ImageDraw

from einops import rearrange, repeat

import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import argparse



def R_nonorm(Z, eps=0.5):
    """Compute the log-determinant term."""
    b = Z.size(-2)
    c = Z.size(-1)
    
    cov = Z.transpose(-2, -1) @ Z
    I = torch.eye(cov.size(-1)).to(Z.device)
    for i in range(len(Z.shape)-2):
        I = I.unsqueeze(0)
    alpha = c/(b*eps)
    
    cov = alpha * cov + I


    out = 0.5 * torch.logdet(cov)
    return out.mean()

def fast_logdet_svd(x):
    """Calculate log determinant using SVD."""
    u, s, v = torch.linalg.svd(x, full_matrices=False)
    return torch.sum(torch.log(s))


def fast_logdet_cholesky(x):
    """Calculate log determinant using Cholesky decomposition."""
    L = torch.linalg.cholesky(x)
    return 2 * torch.sum(torch.log(torch.diag(L)))


class EMAPatchPCANoise(nn.Module):
    def __init__(self, patch_size=4, noise_scale=0.5,kernel='linear',gamma=1.0,alpha=0.995):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None
        self.alpha = alpha

    def forward(self, x,return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)

        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean
            cov = (centered.T @ centered) / (centered.size(0) - 1 + 1e-6)
            if self.ema_cov is None:
                self.ema_cov = cov
            else:
                self.ema_cov = self.ema_cov*self.alpha + cov*(1-self.alpha)
            eig_vals, eig_vecs = torch.linalg.eigh(self.ema_cov+1e-6*torch.eye(self.ema_cov.size(0)).to(self.ema_cov.device))
            # Reverse to get descending order
            eig_vals = eig_vals.flip(0)
            eig_vecs = eig_vecs.flip(1)
            valid_components = torch.sum(eig_vals > 1e-6)
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]

        # Generate PCA-space noise
        # noise_coeff = torch.randn_like(all_patches)  # (B*num_patches_total, C*p*p)
        noise_coeff = torch.randn(all_patches.size(0),valid_components).to(all_patches.device)  # (B*num_patches_total, C*p*p)
        scaled_noise = noise_coeff * (eig_vals.sqrt() * self.noise_scale).unsqueeze(0)
        # scaled_noise = noise_coeff * (self.noise_scale)
        pca_noise = scaled_noise @ eig_vecs.T

        # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise

        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        if return_patches:
            components = all_patches @ eig_vecs
            components = components * torch.sqrt(eig_vals + 1e-8).unsqueeze(0)
            x_components = components.reshape_as(x_patches)
            return noisy_image,x_components
        else:
            return noisy_image



class SpatialPatchPCANoise(nn.Module):
    """Module for applying PCA-based noise to image patches."""
    
    def __init__(self, patch_size=4, noise_scale=0.5, kernel='linear', gamma=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None

    def inverse_transform(self, x_components):
        B, N, C = x_components.shape
        x_components = x_components.reshape(B*N, C)
        return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)
    
    @torch.no_grad()
    def forward(self, x, return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        # # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)

        # x_patches = x_patches.transpose(-2,-1)
        # all_patches = x_patches.reshape(-1, num_patches_h * num_patches_w*C)

        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean

            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]
        
            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs
        
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)

        ema_eig_vecs = self.ema_eig_vecs# @ R
        
        # # scaled_noise = noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0) * self.noise_scale
        # # Calculate energy (sum of eigenvalues)
        # total_energy = self.ema_eig_vals.sum()
        
        # # Calculate relative importance of each eigenvector based on its eigenvalue
        # importance = self.ema_eig_vals / total_energy
        
        # # Apply softmax to get normalized weights that sum to 1
        # # Using temperature parameter to control the distribution sharpness
        # temperature = 0.1 # Higher temperature for smoother distribution
        # softmax_weights = torch.nn.functional.softmax(importance / temperature, dim=0)
        
        # # Weight the noise coefficients by the softmax weights
        # # This gives more emphasis to directions with higher variance
        # weighted_noise_coeff = noise_coeff * softmax_weights.unsqueeze(0)
        
        # # Scale the weighted noise by the noise scale parameter
        # scaled_noise = weighted_noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0) * self.noise_scale

        # linear sample noise_scale from √2 to 0.
        # Linear uniform sampling of noise_scale from √2 to 0
        batch_size = all_patches.size(0)
        # Create a linear distribution from √2 to 0
        max_scale = 2**0.5  # √2
        min_scale = 0
        # Generate random values between 0 and 1, then scale to our range
        random_scales = torch.rand(batch_size, 1, device=all_patches.device)
        # Transform to range [min_scale, max_scale]
        batch_noise_scales = max_scale - random_scales * (max_scale - min_scale)
        # Apply the per-sample noise scale
        scaled_noise = noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0) * batch_noise_scales

        pca_noise = scaled_noise @ ema_eig_vecs.T
        
        
        # # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise
        
        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        if return_patches:
            pca_noise = pca_noise.reshape(B, num_patches_h, num_patches_w, C, p, p)
            pca_noise = pca_noise.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
            pca_noise = pca_noise.reshape(B, C, H, W)
            return noisy_image, pca_noise
        else:
            return noisy_image


class SVDPatchPCANoise(nn.Module):
    """Module for applying PCA-based noise to image patches."""
    
    def __init__(self, patch_size=4, noise_scale=0.5, kernel='linear', gamma=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None

    def inverse_transform(self, x_components):
        B, N, C = x_components.shape
        x_components = x_components.reshape(B*N, C)
        return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)

    def forward(self, x, return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)

        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean

            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]

            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs
        
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
        scaled_noise = noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0)*self.noise_scale
        pca_noise = scaled_noise @ self.ema_eig_vecs.T

        # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise

        # Calculate noise energy per patch
        noise_energy = torch.sum(pca_noise**2, dim=-1)  # L2 norm squared per patch
        
        # Normalize to create weights - can use different normalization strategies
        patch_weights = noise_energy / noise_energy.max()  # Simple min-max normalization
        # print('noise_energy',noise_energy.shape)
        # print('pca_noise',pca_noise.shape)
        # print('patch_weights',patch_weights.shape)

        # Alternative: softmax-based weighting
        # patch_weights = F.softmax(noise_energy / temperature, dim=0)
        
        # Reshape weights to match the original patch dimensions
        patch_weights = patch_weights.reshape(B, -1)
        
        # Store the weights for later use in the model
        self.patch_weights = patch_weights


        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)
        
        if return_patches:
            pca_noise = pca_noise.reshape(B, num_patches_h, num_patches_w, C, p, p)
            pca_noise = pca_noise.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
            pca_noise = pca_noise.reshape(B, C, H, W)
            return noisy_image, pca_noise
        else:
            return noisy_image



class SVDPCANoise(nn.Module):
    """Module for applying PCA-based noise to image patches."""
    
    def __init__(self,noise_scale=0.5, kernel='linear', gamma=1.0):
        super().__init__()

        self.noise_scale = noise_scale
        self.ema_cov = None

    def inverse_transform(self, x_components):
        B, N, C = x_components.shape
        x_components = x_components.reshape(B*N, C)
        return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)

    def forward(self, x, return_patches=False):
        
        B, C = x.shape
        
        # Flatten all patches across batch and spatial dimensions
        all_patches = x  # (B*num_patches_total, C*p*p)
        
        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean

            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]

            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs
        
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
        scaled_noise = noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0)*self.noise_scale
        pca_noise = scaled_noise @ self.ema_eig_vecs.T

        # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x)
        noisy_patches = x + pca_noise

        if return_patches:

            return noisy_patches,pca_noise
        else:
            return noisy_patches



# class SubGraphSVDPatchPCANoise(nn.Module):
#     """Module for applying PCA-based noise to image patches."""
    
#     def __init__(self, patch_size=4, noise_scale=0.5, kernel='linear', gamma=1.0):
#         super().__init__()
#         self.patch_size = patch_size
#         self.noise_scale = noise_scale
#         self.ema_cov = None

#     def inverse_transform(self, x_components):
#         B, N, C = x_components.shape
#         x_components = x_components.reshape(B*N, C)
#         return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)

#     def forward(self, x, return_patches=False):
#         if not self.training:
#             return x

#         B, C, H, W = x.shape
#         p = self.patch_size
#         assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

#         # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
#         x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
#         x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
#         num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)

#         # x_patches = rearrange(x_patches, 'b h w c p1 p2 -> b (h p1) (w p2) c p1 p2')
#         x_patches = rearrange(x_patches, 'b h w c p1 p2 -> b (p1 p2) (c h w) ')
#         D = x_patches.shape[-1]
        
#         # x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        
        

#         # Flatten all patches across batch and spatial dimensions
#         all_patches = x_patches.reshape(-1, D)  # (B*num_patches_total, C*p*p)
#         # print('all_patches',all_patches.shape)
#         # Compute PCA components
#         with torch.no_grad():
#             mean = all_patches.mean(dim=0)
#             centered = all_patches - mean

#             n = centered.size(0)
#             u, s, v = torch.linalg.svd(centered, full_matrices=False)
#             eig_vals = (s**2)/(n-1 + 1e-6)
#             eig_vecs = v.T

#             idx = torch.argsort(eig_vals, descending=True)
#             eig_vals = eig_vals[idx]
#             eig_vecs = eig_vecs[:, idx]

#             valid_components = torch.sum(eig_vals > 1e-6)
#             self.valid_components = valid_components
#             eig_vals = eig_vals[:valid_components]
#             eig_vecs = eig_vecs[:, :valid_components]
            
#             self.ema_eig_vals = eig_vals
#             self.ema_eig_vecs = eig_vecs
        
#         noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
#         scaled_noise = noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0)
#         pca_noise = scaled_noise @ self.ema_eig_vecs.T

#         # Reshape noise and add to original patches
#         pca_noise = pca_noise.reshape_as(x_patches)
#         noisy_patches = x_patches + pca_noise
#         # print('noisy_patches',noisy_patches.shape)
#         # Calculate noise energy per patch
#         noise_energy = torch.sum(pca_noise**2, dim=-1)  # L2 norm squared per patch
        
#         # Normalize to create weights - can use different normalization strategies
#         patch_weights = noise_energy / noise_energy.max()  # Simple min-max normalization
#         # print('noise_energy',noise_energy.shape)
#         # print('pca_noise',pca_noise.shape)
#         # print('patch_weights',patch_weights.shape)

#         # Alternative: softmax-based weighting
#         # patch_weights = F.softmax(noise_energy / temperature, dim=0)
        
#         # Reshape weights to match the original patch dimensions
#         patch_weights = patch_weights.reshape(B, -1)
        
#         # Store the weights for later use in the model
#         self.patch_weights = patch_weights


#         # Reconstruct noisy image from patches
#         # noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        
#         noisy_patches = rearrange(noisy_patches, 'b (p1 p2) (c h w) -> b h w c p1 p2',b=B, p1=p, p2=p, c=C,h=num_patches_h,w=num_patches_w)
#         noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
#         noisy_image = noisy_patches.reshape(B, C, H, W)

#         if return_patches:
#             components = all_patches @ self.ema_eig_vecs
#             components = components * torch.sqrt(self.ema_eig_vals + 1e-8).unsqueeze(0)
#             x_components = components.reshape_as(x_patches)
#             return noisy_image, x_components
#         else:
#             return noisy_image



class SubGraphSVDPatchPCANoise(nn.Module):
    """Module for applying PCA-based noise to image patches."""
    
    def __init__(self, patch_size=4, noise_scale=0.5, kernel='linear', gamma=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None

    def inverse_transform(self, x_components):
        B, N, C = x_components.shape
        x_components = x_components.reshape(B*N, C)
        return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)

    def forward(self, x, return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        # x_patches = x_patches.transpose(-2,-1)
        # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)

        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean

            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]

            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs
        
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
        scaled_noise = noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0)
    

        # After computing eigenvalues and eigenvectors...
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
        

        
        ## make scale factor b*1, range from 0.4~0.7    
        scaling_factor = torch.rand(all_patches.size(0),1) * 0.3 + 0.4
        scaling_factor = scaling_factor.to(all_patches.device)
        latent_noise = noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0)
        
        # print('scaling_factor',scaling_factor)
        # Apply scaling to noise in latent space

        pca_noise = scaling_factor*latent_noise @ self.ema_eig_vecs.T
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise
        
        # Calculate noise energy per patch
        noise_energy = torch.sum(pca_noise**2, dim=-1)  # L2 norm squared per patch
        
        # Normalize to create weights - can use different normalization strategies
        patch_weights = noise_energy / noise_energy.max()  # Simple min-max normalization
        # print('noise_energy',noise_energy.shape)
        # print('pca_noise',pca_noise.shape)
        # print('patch_weights',patch_weights.shape)

        # Alternative: softmax-based weighting
        # patch_weights = F.softmax(noise_energy / temperature, dim=0)
        
        # Reshape weights to match the original patch dimensions
        patch_weights = patch_weights.reshape(B, -1)
        
        # Store the weights for later use in the model
        self.patch_weights = patch_weights

        # noisy_patches = noisy_patches.transpose(-2,-1)
        
        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        if return_patches:
            pca_noise = pca_noise.reshape(B, num_patches_h, num_patches_w, C, p, p) 
            pca_noise = pca_noise.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
            pca_noise = pca_noise.reshape(B, C, H, W)
            return noisy_image, pca_noise
        else:
            return noisy_image




class BernoulliPCANoise(nn.Module):
    def __init__(self, patch_size=4, target_snr=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.target_snr = target_snr
        
    def forward(self, x, return_patches=False):
        if not self.training:
            return x
            
        B, C, H, W = x.shape
        p = self.patch_size
        
        # 提取patches
        x_patches = x.unfold(2, p, p).unfold(3, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)
        

        all_patches = x_patches.reshape(-1, C*p*p)
        # 计算PCA
        with torch.no_grad():
            # 中心化
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean
            
            # SVD分解
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            
            # 计算特征值（奇异值的平方）
            eig_vals = s ** 2
            
            # 计算信号总能量
            signal_power = torch.sum(eig_vals)
            
            # 根据目标SNR计算总噪声能量
            target_noise_power = signal_power / self.target_snr
            
            # 根据特征值大小设计伯努利概率
            # 特征值越大，被置零的概率越大
            probs = eig_vals / eig_vals.max()
            
            # 生成伯努利噪声掩码
            mask = torch.bernoulli(probs).to(x.device)
            
            # 缩放系数，确保总噪声能量符合目标SNR
            scale = torch.sqrt(target_noise_power / (torch.sum(eig_vals * mask) + 1e-8))
            
            # 应用噪声
            noisy_s = s * (1 - mask * scale)
            noisy_s = torch.diag(noisy_s)

            # 重建带噪声的数据
            noisy_patches = u @ noisy_s @ v
            noisy_patches = noisy_patches + mean
            
            
            # # 验证实际SNR（调试用）
            # actual_noise = noisy_patches - all_patches
            # actual_noise_power = torch.mean(actual_noise**2)
            # actual_signal_power = torch.mean(all_patches**2)
            # actual_snr = actual_signal_power / actual_noise_power
            # print(f"Actual SNR: {actual_snr:.3f}")
            
            
            # 重构图像
            noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
            noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)
            noisy_image = noisy_patches.reshape(B, C, H, W)
            
            return noisy_image


class MaxSVDPatchPCANoise(nn.Module):
    """Module for applying PCA-based noise to image patches."""
    
    def __init__(self, patch_size=4, noise_scale=0.5, kernel='linear', gamma=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None

    def inverse_transform(self, x_components):
        B, N, C = x_components.shape
        x_components = x_components.reshape(B*N, C)
        return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)
    
    @torch.no_grad()
    def forward(self, x, return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)

        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean

            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]

            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs
        
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
        scaled_noise = noise_coeff * self.noise_scale * self.ema_eig_vals.sqrt().unsqueeze(0)
        pca_noise = scaled_noise @ self.ema_eig_vecs.T

        # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise

        # Calculate noise energy per patch
        noise_energy = torch.sum(pca_noise**2, dim=-1)  # L2 norm squared per patch
        
        # Normalize to create weights - can use different normalization strategies
        patch_weights = noise_energy / noise_energy.max()  # Simple min-max normalization
        # print('noise_energy',noise_energy.shape)
        # print('pca_noise',pca_noise.shape)
        # print('patch_weights',patch_weights.shape)

        # Alternative: softmax-based weighting
        # patch_weights = F.softmax(noise_energy / temperature, dim=0)
        
        # Reshape weights to match the original patch dimensions
        patch_weights = patch_weights.reshape(B, -1)
        
        # Store the weights for later use in the model
        self.patch_weights = patch_weights


        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        

        if return_patches:
            pca_noise = pca_noise.reshape(B, num_patches_h, num_patches_w, C, p, p)
            pca_noise = pca_noise.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
            pca_noise = pca_noise.reshape(B, C, H, W)
            return noisy_image, pca_noise
        else:
            return noisy_image



class MultiScaleSVDPatchPCANoise(nn.Module):
    """Module for applying PCA-based noise to image patches."""
    
    def __init__(self, patch_size=4, noise_scale=1.5, kernel='linear', gamma=1.0,steps = 40):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None
        self.steps = steps



    def inverse_transform(self, x_components):
        B, N, C = x_components.shape
        x_components = x_components.reshape(B*N, C)
        return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)
    
    @torch.no_grad()
    def forward(self, x, return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        N = num_patches_h * num_patches_w
        D = C*p*p
        
        # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)

        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean

            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]

            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs

        sigmas = torch.linspace(0.002, self.noise_scale, self.steps).to(x.device)
        indices = torch.randint(
            0, self.steps - 1, (B,), device=x.device
        )
        sigma = sigmas[indices].view(-1,1,1)
        
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
        
        scaled_noised_coeff = noise_coeff.reshape(B,N,self.valid_components) * sigma
        scaled_noised_coeff = scaled_noised_coeff.reshape(B*N,self.valid_components)
        scaled_noise = scaled_noised_coeff * self.ema_eig_vals.sqrt().unsqueeze(0)
        pca_noise = scaled_noise @ self.ema_eig_vecs.T

        # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise

        # Calculate noise energy per patch
        noise_energy = torch.sum(pca_noise**2, dim=-1)  # L2 norm squared per patch
        
        # Normalize to create weights - can use different normalization strategies
        patch_weights = noise_energy / noise_energy.max()  # Simple min-max normalization
        # print('noise_energy',noise_energy.shape)
        # print('pca_noise',pca_noise.shape)
        # print('patch_weights',patch_weights.shape)

        # Alternative: softmax-based weighting
        # patch_weights = F.softmax(noise_energy / temperature, dim=0)
        
        # Reshape weights to match the original patch dimensions
        patch_weights = patch_weights.reshape(B, -1)
        
        # Store the weights for later use in the model
        self.patch_weights = patch_weights


        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        

        if return_patches:
            pca_noise = pca_noise.reshape(B, num_patches_h, num_patches_w, C, p, p)
            pca_noise = pca_noise.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
            pca_noise = pca_noise.reshape(B, C, H, W)
            return noisy_image, pca_noise
        else:
            return noisy_image



class UniformSVDPatchPCANoise(nn.Module):
    """Module for applying PCA-based noise to image patches."""
    
    def __init__(self, patch_size=4, noise_scale=0.5, kernel='linear', gamma=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None

    def inverse_transform(self, x_components):
        B, N, C = x_components.shape
        x_components = x_components.reshape(B*N, C)
        return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)
    
    @torch.no_grad()
    def forward(self, x, return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)

        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean

            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]

            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs
        
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
        noise_coeff = (noise_coeff-0.5)*2
        
        scaled_noise = noise_coeff * self.noise_scale * self.ema_eig_vals.sqrt().unsqueeze(0)
        pca_noise = scaled_noise @ self.ema_eig_vecs.T

        # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise

        # Calculate noise energy per patch
        noise_energy = torch.sum(pca_noise**2, dim=-1)  # L2 norm squared per patch
        
        # Normalize to create weights - can use different normalization strategies
        patch_weights = noise_energy / noise_energy.max()  # Simple min-max normalization
        # print('noise_energy',noise_energy.shape)
        # print('pca_noise',pca_noise.shape)
        # print('patch_weights',patch_weights.shape)

        # Alternative: softmax-based weighting
        # patch_weights = F.softmax(noise_energy / temperature, dim=0)
        
        # Reshape weights to match the original patch dimensions
        patch_weights = patch_weights.reshape(B, -1)
        
        # Store the weights for later use in the model
        self.patch_weights = patch_weights


        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        

        if return_patches:
            pca_noise = pca_noise.reshape(B, num_patches_h, num_patches_w, C, p, p)
            pca_noise = pca_noise.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
            pca_noise = pca_noise.reshape(B, C, H, W)
            return noisy_image, pca_noise
        else:
            return noisy_image


class WhitenedSVDPatchPCANoise(nn.Module):
    """Module for applying PCA-based noise to image patches."""
    
    def __init__(self, patch_size=4, noise_scale=0.5, kernel='linear', gamma=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None

    def inverse_transform(self, x_components):
        B, N, C = x_components.shape
        x_components = x_components.reshape(B*N, C)
        return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)

    def forward(self, x, return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)

        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean

            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]

            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs
        
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
        scaled_noise = noise_coeff * self.noise_scale 
        pca_noise = scaled_noise @ self.ema_eig_vecs.T
        
        # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise

        # Calculate noise energy per patch
        noise_energy = torch.sum(pca_noise**2, dim=-1)  # L2 norm squared per patch
        
        # Normalize to create weights - can use different normalization strategies
        patch_weights = noise_energy / noise_energy.max()  # Simple min-max normalization
        # print('noise_energy',noise_energy.shape)
        # print('pca_noise',pca_noise.shape)
        # print('patch_weights',patch_weights.shape)

        # Alternative: softmax-based weighting
        # patch_weights = F.softmax(noise_energy / temperature, dim=0)
        
        # Reshape weights to match the original patch dimensions
        patch_weights = patch_weights.reshape(B, -1)
        
        # Store the weights for later use in the model
        self.patch_weights = patch_weights


        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        if return_patches:
            components = all_patches @ self.ema_eig_vecs
            components = components * torch.sqrt(self.ema_eig_vals + 1e-8).unsqueeze(0)
            x_components = components.reshape_as(x_patches)
            return noisy_image, x_components
        else:
            return noisy_image


class KernelPCA(nn.Module):
    """核PCA模块，支持分批次近似计算与噪声添加"""
    def __init__(self, n_components=768, kernel='rbf', gamma=1.0, n_anchors=8192, batch_size=64):
        super().__init__()
        self.n_components = n_components    # 主成分数量
        self.kernel = kernel                # 核函数类型（支持'rbf', 'poly'）
        self.gamma = gamma                  # RBF核参数
        self.n_anchors = n_anchors          # 锚点数量
        self.batch_size = batch_size        # 批次大小
        
        # 注册缓冲区存储锚点和核矩阵
        self.register_buffer('anchors', None)         # 锚点数据 (n_anchors, d)
        self.register_buffer('K_mm', None)           # 锚点核矩阵 (n_anchors, n_anchors)
        self.register_buffer('alpha', None)          # 投影系数 (n_anchors, n_components)
        self.register_buffer('mean_anchor', None)    # 锚点均值，用于中心化
        self.register_buffer('components', None)     # 主成分向量 (d, n_components) - 用于线性核

    @torch.no_grad()
    def _kernel_func(self, X, Y):
        """计算核矩阵 (RBF或多项式核)"""
        if self.kernel == 'rbf':
            dist = torch.cdist(X, Y, p=2)**2
            return torch.exp(-self.gamma * dist)

        elif self.kernel == 'poly':
            return (X @ Y.T)**2
        elif self.kernel =='linear':
            return X @ Y.T
        else:
            raise ValueError("Unsupported kernel type")

    def _update_anchors(self, batch):
        """动态更新锚点集合 (FIFO策略)"""
        # shuffle batch
        batch = batch[torch.randperm(batch.size(0))]
        if self.anchors is None:
            # 初始化锚点为当前批次前n_anchors个样本            
            self.anchors = batch[:min(self.n_anchors, batch.size(0))].detach()
            # 计算并存储锚点均值
            self.mean_anchor = self.anchors.mean(dim=0, keepdim=True)
        else:
            # 合并新旧锚点，保留最新的n_anchors个
            combined = torch.cat([self.anchors, batch], dim=0)
            self.anchors = combined[-self.n_anchors:].detach()
            # 更新锚点均值
            self.mean_anchor = self.anchors.mean(dim=0, keepdim=True)

    def _compute_nystrom_components(self):
        """计算Nyström近似的主成分"""
        # 对于线性核，我们可以直接计算PCA
        if self.kernel == 'linear':
            # 中心化锚点数据
            centered_anchors = self.anchors - self.mean_anchor
            
            # 计算协方差矩阵
            cov = centered_anchors.T @ centered_anchors / (self.anchors.size(0) - 1)
            
            # 计算特征分解
            eig_vals, eig_vecs = torch.linalg.eigh(cov)
            
            idx = torch.argsort(eig_vals, descending=True)
            
            valid_components = torch.sum(eig_vals > 1e-6)
            eig_vals = eig_vals[idx][:valid_components]
            eig_vecs = eig_vecs[:, idx][:, :valid_components]
            
            # 存储主成分向量和特征值
            self.components = eig_vecs
            self.eigenvalues = eig_vals.clamp(min=1e-8)
            
            # 同时保持与非线性核一致的接口
            K_mm = self._kernel_func(self.anchors, self.anchors)
            self.K_mm = K_mm
            
            # 中心化核矩阵
            ones_m = torch.ones_like(K_mm) / self.n_anchors
            K_mm_centered = K_mm - ones_m @ K_mm - K_mm @ ones_m + ones_m @ K_mm @ ones_m
            
            # 计算特征分解
            eig_vals_k, eig_vecs_k = torch.linalg.eigh(K_mm_centered)
            idx = torch.argsort(eig_vals_k, descending=True)
            eig_vals_k = eig_vals_k[idx][:valid_components]
            eig_vecs_k = eig_vecs_k[:, idx][:, :valid_components]
            
            # 正则化防止数值问题
            eig_vals_k = torch.clamp(eig_vals_k, min=1e-8)
            self.alpha = eig_vecs_k / torch.sqrt(eig_vals_k).unsqueeze(0)
        else:
            # 对于非线性核，使用原始的Nyström方法
            K_mm = self._kernel_func(self.anchors, self.anchors)  
            self.K_mm = K_mm
            
            # 中心化核矩阵
            ones_m = torch.ones_like(K_mm) / self.n_anchors
            K_mm_centered = K_mm - ones_m @ K_mm - K_mm @ ones_m + ones_m @ K_mm @ ones_m
            
            # 计算特征分解
            eig_vals, eig_vecs = torch.linalg.eigh(K_mm_centered)
            idx = torch.argsort(eig_vals, descending=True)
            valid_components = torch.sum(eig_vals > 1e-6)
            eig_vals = eig_vals[idx][:valid_components]
            eig_vecs = eig_vecs[:, idx][:, :valid_components]
            
            # 正则化防止数值问题
            eig_vals = torch.clamp(eig_vals, min=1e-8)
            self.alpha = eig_vecs / torch.sqrt(eig_vals).unsqueeze(0)
            self.eigenvalues = eig_vals

    @torch.no_grad()
    def forward(self, x, noise_scale=(3**0.5)):
        """
        输入: x - 图像张量 (B, C, H, W)
        输出: x_recon - 加噪后重建图像 (B, C, H, W)
        """
        B, C = x.shape
        
        # 1. 动态更新锚点集合
        x_flatten = x
        self._update_anchors(x_flatten)

        # 2. 计算Nyström主成分 (每批次更新)
        self._compute_nystrom_components()
        
        # 3. PCA变换与重建
        if self.kernel == 'linear':
            # 对于线性核，我们可以直接使用PCA变换和重建
            # 中心化输入数据
            x_centered = x_flatten - self.mean_anchor
            
            # 计算主成分空间中的表示
            z = x_centered @ self.components  # (B, n_components)
            
            # 可选：在主成分空间中添加噪声
            if noise_scale > 0 and self.training:
                noise = torch.randn_like(z) * noise_scale * torch.sqrt(self.eigenvalues)
                z = z + noise
            
            # 重建原始空间
            x_recon_flatten = z @ self.components.T + self.mean_anchor
            
            return x_recon_flatten
        else:
            # 对于非线性核，使用Nyström方法
            m = self.K_mm.shape[0]
            
            # 计算输入与锚点的核矩阵
            K_nm = self._kernel_func(x_flatten, self.anchors)  # (B, m)
            
            # 中心化核矩阵
            ones_n = torch.ones(B, m).to(x.device) / m
            K_nm_centered = K_nm - ones_n @ self.K_mm - K_nm @ (self.K_mm.sum(dim=1, keepdim=True) / m)
            

            
            # 投影到核空间主成分
            z = K_nm_centered @ self.alpha  # (B, k)
            
            # 可选：在主成分空间中添加噪声
            if noise_scale > 0 and self.training:
                noise = torch.randn_like(z) * noise_scale * torch.sqrt(self.eigenvalues)
                z = z + noise

            # 计算重建系数
            beta = z @ self.alpha.T  # (B, m)
            
            # 重建原始空间
            x_recon_flatten = beta @ self.anchors  # (B, d)

            return x_recon_flatten
    
    
    def rbf_preimage(self,z, anchors, gamma):
        """Better pre-image estimation for RBF kernel"""
        # Initialize with weighted average of anchors
        x_init = (z @ self.alpha.T) @ anchors
        
        # Iterative refinement
        x_solution = x_init.clone()
        for i in range(10):  # Fixed number of iterations
            # Compute weights based on kernel distances
            weights = torch.exp(-gamma * torch.sum((anchors - x_solution.unsqueeze(1))**2, dim=2))
            weights = weights / weights.sum(dim=1, keepdim=True)
            
            # Update solution
            x_new = weights @ anchors
            x_solution = x_new
            
        return x_solution
        
    def extra_repr(self):
        return f"kernel={self.kernel}, anchors={self.n_anchors}, components={self.n_components}"



class BatchwiseKernelPatchPCANoise(nn.Module):
    """Module for applying kernel PCA-based noise to image patches using all components."""
    
    def __init__(self, patch_size=4, noise_scale=0.5, kernel='linear', gamma=2):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.kernel = kernel
        self.gamma = gamma
        # self.kpca = SKLKernelPCA(fit_inverse_transform=True,kernel=kernel,gamma=self.gamma)
        self.kpca = KernelPCA(kernel=kernel,gamma=self.gamma)
        # self.kpca.eval()

    @torch.no_grad()
    def _kernel_func(self, X, Y):
        """Compute kernel matrix between X and Y."""
        if self.kernel == 'rbf':
            dist = (X.unsqueeze(1) - Y.unsqueeze(2))**2
            dist = dist.sum(dim=-1)
            return torch.exp(-self.gamma * dist)
        elif self.kernel == 'poly':
            return (X @ Y.transpose(-2, -1))**2
        else:  # default to linear kernel
            return X @ Y.transpose(-2, -1)

    def _compute_kernel_pca(self, X,K):
        """Compute kernel PCA components for input data."""
        n_samples = X.size(0)
        
        # Compute and center kernel matrix

        ones_n = torch.ones(n_samples, n_samples).to(X.device) / n_samples
        K_centered = K - ones_n @ K - K @ ones_n + ones_n @ K @ ones_n
        
        # Compute eigendecomposition
        eig_vals, eig_vecs = torch.linalg.eigh(K_centered)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(eig_vals, descending=True)
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        # Keep only valid components (positive eigenvalues)
        valid_components = torch.sum(eig_vals > 1e-6)
        eig_vals = eig_vals[:valid_components]
        eig_vecs = eig_vecs[:, :valid_components]
        
        # Normalize eigenvectors
        eig_vecs = eig_vecs / torch.sqrt(eig_vals + 1e-8)
        
        return eig_vals, eig_vecs, valid_components


    def forward(self, x, return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        noisy_patches = []
        patch_weights_list = []
        
        # x_patches = x_patches.reshape(1,-1,x_patches.shape[-1])
        # BB,N,_ = x_patches.shape
        # b_K = self._kernel_func(x_patches, x_patches)
        
        x_patches = x_patches.reshape(-1,x_patches.shape[-1])
        
        # x_patches = x_patches.reshape(B,-1)
        # x_patches = x_patches.detach().cpu().numpy()
        
        noisy_patches = self.kpca(x_patches)
        
        
        # Reconstruct image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        if return_patches:
            components = K @ eig_vecs
            components = components * torch.sqrt(eig_vals + 1e-8).unsqueeze(0)
            x_components = components.reshape_as(x_patches[b])
            return noisy_image, x_components
        else:
            return noisy_image
    

def R_nonorm(Z, eps=0.5, if_fast=False):
    """Compute the log-determinant term."""
    b = Z.size(-2)
    c = Z.size(-1)
    
    cov = Z.transpose(-2, -1) @ Z
    I = torch.eye(cov.size(-1)).to(Z.device)
    for i in range(len(Z.shape)-2):
        I = I.unsqueeze(0)
    alpha = c/(b*eps)
    
    cov = alpha * cov + I

    if if_fast:
        out = 0.5 * fast_logdet_cholesky(cov)
    else:
        out = 0.5 * torch.logdet(cov)
    return out.mean()


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class PermutePatch(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, psz):
        self.psz = psz

    def __call__(self, img):
        imgs = []
        imgwidth, imgheight = img.size
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                box = (j, i, j+self.psz, i+self.psz)
                imgs.append(img.crop(box))
        random.shuffle(imgs)
        new_img = Image.new('RGB', (imgwidth, imgheight))
        k = 0
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                new_img.paste(imgs[k], (j, i))
                k += 1
        return new_img

class HideAndSeek(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, ratio, psz):
        self.ratio = ratio
        self.psz = psz

    def __call__(self, img):
        imgwidth, imgheight = img.size 
        numw, numh = imgwidth // self.psz, imgheight // self.psz
        mask_num = int(numw * numh * self.ratio)
        mask_patch = np.random.choice(np.arange(numw * numh), mask_num, replace=False)
        mask_w, mask_h = mask_patch % numh, mask_patch // numh
        # img.save('test1.png')
        draw = ImageDraw.Draw(img)
        for mw, mh in zip(mask_w, mask_h):
            draw.rectangle((mw * self.psz, 
                            mh * self.psz,
                            (mw + 1) * self.psz,
                            (mh + 1) * self.psz), fill="black")
        # img.save('test2.png')
        return img

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        return
    elif pretrained_weights == 'download':
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
            return
    elif pretrained_weights == 'supervised':
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "deit_small_patch16_224-cd65a155.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "deit_base_patch16_224-b5f2ef4d.pth"
        if url is not None:
            print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/" + url)
            msg = model.load_state_dict(state_dict['model'], strict=False)
            print('Supervised weights found at {} and loaded with msg: {}'.format(url, msg))
            return
    print("There is no reference weights available for this model => We use random weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor



class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, 
                **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx: end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx: end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_



class GaussianNoiseLayer(nn.Module):
    def __init__(self, mean=0.0, std=0.0):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        # 添加高斯噪声
        if self.training:
            noise = self.mean + self.std * torch.randn(x.size()).to(x)
            return x + noise
        return x

class MultiCropCtrlWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone,decoder=None,head=None,decoder_res=None,noise=0.0):
        super(MultiCropCtrlWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        if decoder is None:
            self.decoder = nn.Identity()
        else:
            self.decoder = decoder
        
        if decoder_res is None:
            self.decoder_res = None
        else:
            self.decoder_res = decoder_res
        

        self.noise_layer = GaussianNoiseLayer(0,noise) if noise >0 else nn.Identity()

    def forward_encoder(self,x,mask=None):
        return self.backbone(x)
    
    def forward_decoder(self,x):
        return self.decoder(x)
    
    def forward(self, x, mask=None, return_backbone_feat=False,detach_encoder=False, z_extra=None,dif_sampler=None,comp=False,forward_decoder=True,
                **kwargs):
        if dif_sampler is not None:
            B = x.shape[0]
            t = torch.randint(low=0, high=1000, size=(B,)).to(x).long()
            x_noised,noise = dif_sampler.make_noisy(x, t)
        else:
            x_noised =x
        z = self.backbone(x_noised,mask=mask)
        if forward_decoder is False:
            
            z_o = z
            z = self.head(z)
            
            return z,z,x,z_o,z
        
        if z_extra is not None:
            x_hat = self.decoder(self.noise_layer(z_extra))
        else:
            
            # if detach_encoder:
            #     x_hat = self.decoder(self.noise_layer(z.detach()))
            # else:
            x_hat = self.decoder(self.noise_layer(z))
        


        if dif_sampler is not None:
            x_hat_noise,_,mu,log_var = x_hat
            x_hat = dif_sampler.denoise_at_t(x_noised, timestep=t, t=0,epsilon_pred=x_hat_noise)
        else:
            x_hat,_,mu,log_var = x_hat
            noise = None
            x_hat_noise = None
        
        # x_hat = x_noised
        if detach_encoder:
            z_hat = self.backbone(x_hat.detach(),mask=mask)
        else:
            if comp:        
                mask = ~mask
            z_hat = self.backbone(x_hat,mask=mask)
        
        if mask is not None:
            mask_compl = ~mask
        else:
            mask_compl = None
        
        # z_compl = self.backbone(x.detach(),mask=mask_compl)
        # z_compl_hat = self.backbone(x_hat.detach(),mask=mask_compl)

        # input should be clean input image x, as x hat is reconstruction of clean input x
        if self.decoder_res is not None:
            
            # z_compl = self.backbone(x.detach(),mask=mask_compl)
            z_compl_hat = self.backbone(x_hat.detach(),mask=mask_compl)

            res_target = (x-x_hat).detach()
            res_pred,_,_,_ = self.decoder_res(z_compl_hat)
            z = self.head(z)
            z_hat = self.head(z_hat)
            
            return z,z_hat,x_hat,res_target,res_pred
        
        z_o = z
        z = self.head(z)
        z_hat = self.head(z_hat)

        return z,z_hat,x_hat,z_o,z
        



def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]




def get_params_groups_by_block(model,blocks=['backbone']):
    regularized = []
    not_regularized = []
    print('=='*20)
    for name, param in model.named_parameters():
        if (not param.requires_grad):
            continue
        
        is_param_in_blk = 0
        for blk in blocks:
            if blk in name:
                is_param_in_blk=1
        
        if is_param_in_blk==0:
            continue
        
        # we do not regularize biases nor Norm parameters
        print('name',name)
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]



def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T
        
        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)
        
        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)

        out = torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)
        return out



def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs

def rbf_preimage(z, anchors, gamma):
    """Better pre-image estimation for RBF kernel"""
    # Initialize with weighted average of anchors
    x_init = (z @ self.alpha.T) @ anchors
    
    # Iterative refinement
    x_solution = x_init.clone()
    for i in range(10):  # Fixed number of iterations
        # Compute weights based on kernel distances
        weights = torch.exp(-gamma * torch.sum((anchors - x_solution.unsqueeze(1))**2, dim=2))
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Update solution
        x_new = weights @ anchors
        x_solution = x_new
        
    return x_solution


def generate_random_rotation_matrix(n, theta_max=0.1,theta_min=0.00):
    """Generate a random rotation matrix with small angles.
    Args:
        n: dimension of the matrix
        theta_max: maximum rotation angle in radians (default 0.1 ≈ 5.7 degrees)
    """
    # Generate random skew-symmetric matrix
    random_state = torch.rand(n, n)
    skew = random_state - random_state.t()
    
    # Randomly select a value between theta_min and theta_max
    theta_range = theta_max - theta_min
    random_theta = theta_min + theta_range * torch.randn(1).item()
    # Scale the rotation to be small
    skew = skew * (random_theta / torch.max(torch.abs(skew)))
    
    # Convert to rotation matrix using Cayley transform
    I = torch.eye(n).to(skew.device)
    R = (I - skew) @ torch.inverse(I + skew)
    
    return R




class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number,img_size=(224,224)):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.img_size = img_size

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(int(self.img_size[0]*96/224), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
