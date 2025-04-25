import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import trunc_normal_
import os
os.environ['MPLBACKEND'] = 'Agg'  # Set this before importing matplotlib
import matplotlib.pyplot as plt
from utils import cosine_scheduler
from torchvision.utils import make_grid
import torch.utils.checkpoint
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
import argparse
from pathlib import Path
import math
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.optim import create_optimizer_v2
from karras_sampler import KarrasSampler,get_sigmas_karras,EDMPrecondMae
from einops import rearrange
import seaborn as sns
import os
from typing import Callable

class EDMPrecondNoProp(nn.Module):
    """
    EDM preconditioner for model outputs
    """
    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = 0.5
    
    def forward(self, x, noised_image, mask, ids_restore,sigma, **kwargs) -> torch.Tensor:
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        x_in = c_in * noised_image
        
        F_x = self.inner_model(x, x_in, mask, ids_restore)

        D_x = c_skip * self.inner_model.patchify(noised_image) + c_out * F_x
        
        return D_x



class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3,
                 decoder_embed_dim=96, decoder_depth=4, decoder_num_heads=3,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embed_dim = embed_dim
        self.patch_size = patch_size    
        self.num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        

        self.precond = EDMPrecondMae(self)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_decoder = PatchEmbed(img_size, patch_size, in_chans, decoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.initialize_weights()
        self.sampler = KarrasSampler(
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            num_steps=40,
            sampling='linear'
        )

    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                          int(self.patch_embed.num_patches**.5), 
                                          cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 
                                                   int(self.patch_embed.num_patches**.5), 
                                                   cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize tokens and other parameters
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_feature(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)  # Enable gradient checkpointing
            else:
                x = blk(x)
        x = self.norm(x)
        return x

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)  # Enable gradient checkpointing
            else:
                x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def sample(self, x):
        return self.sampler.sample(x)

    def forward_decoder(self, x,noised_image, mask,ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x_dec = self.patch_embed_decoder(noised_image)

        x_ = (1-mask.unsqueeze(-1)) * x_ + mask.unsqueeze(-1) * x_dec

        
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)  # Enable gradient checkpointing
            else:
                x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # Remove CLS token

        return x

    def forward_loss(self, pred, target, mask):
        target = self.patchify(target)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        
        noised_image,sigma = self.sampler.add_noise(imgs)
        
        snrs = sigma**-2       
        weightings = snrs + 1.0
        # use dummy weightings
        weightings = torch.ones_like(weightings)

        
        pred = self.precond(latent, noised_image, mask, ids_restore,sigma)
        loss = self.forward_loss(pred, imgs, mask)
        return loss, pred, mask
    
    
    def denoise(self, noised_image, latent, mask,ids_restore,sigma):
        
        pred = self.precond(latent, noised_image, mask, ids_restore,sigma)
        pred = self.unpatchify(pred)
        return pred
    

    def get_attention_maps(self, x, layer_idx=-1):
        """
        Get attention maps from a specific transformer layer
        Args:
            x: Input tensor
            layer_idx: Index of transformer layer to visualize (-1 for last layer)
        Returns:
            attention_maps: [B, H, N, N] attention weights
        """
        B = x.shape[0]
        
        # Get patches
        x = self.patch_embed(x)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, 1:, :]
        
        # Add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Pass through transformer blocks until target layer
        target_block = self.blocks[layer_idx]
        
        # Get attention weights from target block
        with torch.no_grad():
            # Forward pass until attention
            attn = target_block.attn
            qkv = attn.qkv(x)
            qkv = rearrange(qkv, 'b n (h d qkv) -> qkv b h n d', h=attn.num_heads, qkv=3)
            q, k, v = qkv[0], qkv[1], qkv[2]   # b h n d
            
            # Calculate attention weights
            attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
            attn_weights = attn_weights.softmax(dim=-1)  # b h n n
            
        return attn_weights

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sinusoidal position embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def visualize_attention(model, images, save_path='attention_maps', layer_idx=-1):
    """Visualize attention maps for given images"""
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get attention weights
        attn_weights = model.get_attention_maps(images, layer_idx=layer_idx)
        
        # Average over heads
        attn_weights = attn_weights.mean(dim=1)  # [B, N, N]
        
        # Plot attention maps
        n_images = min(4, images.shape[0])
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        
        for i in range(n_images):
            # Plot original image
            img = images[i].cpu()
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0., 1.)
            axes[0, i].imshow(img.permute(1, 2, 0))
            axes[0, i].axis('off')
            axes[0, i].set_title('Original Image')
            
            # Plot attention map
            attn = attn_weights[i].cpu()
            sns.heatmap(attn.numpy(), ax=axes[1, i], cmap='viridis')
            axes[1, i].set_title('Attention Map')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'attention_map_layer_{layer_idx}.png'))
        plt.close()
        
        return attn_weights




class DenoisingBlock(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,embed_dim=192, decoder_embed_dim=96,decoder_depth=4,decoder_num_heads=3,mlp_ratio=4,norm_layer=nn.LayerNorm,use_checkpoint=False):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed_decoder = PatchEmbed(img_size, patch_size, in_chans, decoder_embed_dim)
        self.use_checkpoint = use_checkpoint
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))    
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)


        self.blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize position embeddings
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 
                                                   int(self.patch_embed_decoder.num_patches**.5), 
                                                   cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.patch_embed_decoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward(self, x,noised_image, mask,ids_restore):

        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        x_dec = self.patch_embed_decoder(noised_image)

        x_ = (1-mask.unsqueeze(-1)) * x_ + mask.unsqueeze(-1) * x_dec
        
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)  # Enable gradient checkpointing
            else:
                x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # Remove CLS token

        return x


class MaskedDenoisingVit():
    def __init__(self, T=4, 
                 args = None,
                 device = 'cuda',
                 trainset = None,
                 train_loader = None,
                 ):
        self.T = T
        self.device = args.device
        self.args = args
        
        self.sampler = KarrasSampler(
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            num_steps=T,
            sampling='linear')
        
        # Get EDM sigmas for training
        self.sigmas = get_sigmas_karras(
            T,
            sigma_min=self.sampler.sigma_min,
            sigma_max=self.sampler.sigma_max,
            rho=self.sampler.rho,
            device=device
        )
        
        # Models initialization
        self.encoder = MaskedAutoencoderViT(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_depth=args.decoder_depth,
            decoder_num_heads=args.decoder_num_heads,
            mlp_ratio=args.mlp_ratio,
            norm_layer=nn.LayerNorm,
            norm_pix_loss=False,
            use_checkpoint=args.use_checkpoint
        ).to(device)
        
        self.trainset = trainset
        self.train_loader = train_loader

        self.decoders = nn.ModuleList([])

        self.shared_decoder = DenoisingBlock(
                img_size=args.img_size, 
                patch_size=args.patch_size, 
                in_chans=3, 
                embed_dim=args.embed_dim, 
                decoder_embed_dim=args.decoder_embed_dim,
                decoder_depth=args.decoder_depth,
                decoder_num_heads=args.decoder_num_heads,
                mlp_ratio=args.mlp_ratio,
                norm_layer=nn.LayerNorm,
                use_checkpoint=args.use_checkpoint
            ).to(device)
        self.shared_decoder = EDMPrecondNoProp(self.shared_decoder)
        
        for t in range(T):
            blk = DenoisingBlock(
                img_size=args.img_size, 
                patch_size=args.patch_size, 
                in_chans=3, 
                embed_dim=args.embed_dim, 
                decoder_embed_dim=args.decoder_embed_dim,
                decoder_depth=args.decoder_depth,
                decoder_num_heads=args.decoder_num_heads,
                mlp_ratio=args.mlp_ratio,
                norm_layer=nn.LayerNorm,
                use_checkpoint=args.use_checkpoint
            ).to(device)
            blk = EDMPrecondNoProp(blk)
            self.decoders.append(blk)

        # Optimizers
        self.encoder_optimizer = create_optimizer_v2(
            self.encoder,
            opt=args.opt,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            eps=args.opt_eps,
            betas=tuple(args.opt_betas) if args.opt_betas else (0.5, 0.999),  # Provide default tuple
        )
        self.decoder_optimizers = []
        
        for decoder in self.decoders:
            self.decoder_optimizers.append(create_optimizer_v2(
                decoder,
                opt=args.opt,
                lr=args.lr,
                weight_decay=args.weight_decay,
            ))
            

        self.lr_scheduler = cosine_scheduler(base_value=args.lr, final_value=args.min_lr, epochs=args.epochs, niter_per_ep=len(train_loader), warmup_epochs=args.warmup_epochs, start_warmup_value=args.start_warmup_value)
        

    def forward_loss(self,pred,target,mask):
        target = self.encoder.patchify(target)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss


    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Load encoder state
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            
            # Load decoders state
            for i, decoder in enumerate(self.decoders):
                decoder.load_state_dict(checkpoint[f'decoder_{i}_state_dict'])
            
            # Load optimizer states
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
            for i, opt in enumerate(self.decoder_optimizers):
                opt.load_state_dict(checkpoint[f'decoder_optimizer_{i}_state_dict'])
            
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Resuming from epoch {epoch} with loss {loss:.4f}")
            return epoch + 1
        return 0

    def save_checkpoint(self, epoch, save_dir, loss=None):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'loss': loss
        }
        
        # Save decoder states and optimizers
        for i, decoder in enumerate(self.decoders):
            checkpoint[f'decoder_{i}_state_dict'] = decoder.state_dict()
            checkpoint[f'decoder_optimizer_{i}_state_dict'] = self.decoder_optimizers[i].state_dict()
        
        # Save epoch checkpoint
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)
        
        print(f"Saved checkpoint to {epoch_path}")
        
        # If this is the best loss so far, save as best model
        if loss is not None:
            if not hasattr(self, 'best_loss') or loss < self.best_loss:
                self.best_loss = loss
                best_path = os.path.join(save_dir, 'model_best.pth')
                torch.save(checkpoint, best_path)
                print(f"New best model saved to {best_path}")

    

    def train_model(self):
        # Resume from checkpoint if specified
        start_epoch = 0
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume)
        print(f"Starting training from epoch {start_epoch}")

        x,_ = next(iter(self.train_loader))
        x = x.to(self.device)
        visualize_reconstruction(self,x[:8],save_path='reconstructions',epoch=start_epoch,args=self.args)
        
        # Initialize AMP scaler if using mixed precision
        scaler = GradScaler() if self.args.use_amp else None
        
        for epoch in range(start_epoch, self.args.epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i, (x, _) in enumerate(self.train_loader):
                it = i + epoch * len(self.train_loader)
                lr = self.lr_scheduler[it]
                self.encoder_optimizer.param_groups[0]['lr'] = lr
                for opt in self.decoder_optimizers:
                    opt.param_groups[0]['lr'] = lr

                x = x.to(self.device)
                u_y = x

                # Use autocast for mixed precision training or nullcontext if not using AMP
                amp_context = autocast() if self.args.use_amp else nullcontext()
                
                with amp_context:
                    # Forward diffusion using EDM noise schedule
                    z = [u_y]
                    for t in range(1, self.T + 1):
                        eps = torch.randn_like(u_y)
                        sigma = self.sigmas[t-1]
                        z_t = u_y + eps * sigma
                        z.append(z_t)

                    # Extract image features once
                    x_features, mask, ids_restore = self.encoder.forward_encoder(x,self.args.mask_ratio)
                    
                    # Train MLPs independently
                    losses = []
                    for t in range(self.T):
                        # Each MLP tries to denoise from its specific noise level
                        sigma = self.sigmas[t]
                        blk = self.decoders[t]
                        # blk = self.shared_decoder
                        u_hat = blk(x_features, z[t+1].detach(), mask, ids_restore,sigma)
                        loss = self.forward_loss(u_hat,u_y,mask)
                        losses.append(loss)

                    # Optimize all models
                    total_loss = sum(losses)

                # Optimization step with AMP support
                self.encoder_optimizer.zero_grad()
                for opt in self.decoder_optimizers:
                    opt.zero_grad()

                if self.args.use_amp:
                    # Use AMP's scaled backward pass
                    scaler.scale(total_loss).backward()
                    
                    # Unscale before gradient clipping (if used)
                    if self.args.clip_grad is not None:
                        scaler.unscale_(self.encoder_optimizer)
                        for opt in self.decoder_optimizers:
                            scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.clip_grad)
                        for decoder in self.decoders:
                            torch.nn.utils.clip_grad_norm_(decoder.parameters(), self.args.clip_grad)
                    
                    # Step with scaler
                    scaler.step(self.encoder_optimizer)
                    for opt in self.decoder_optimizers:
                        scaler.step(opt)
                    scaler.update()
                else:
                    # Regular backward pass
                    total_loss.backward()
                    
                    # Gradient clipping if specified
                    if self.args.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.clip_grad)
                        for decoder in self.decoders:
                            torch.nn.utils.clip_grad_norm_(decoder.parameters(), self.args.clip_grad)
                    
                    # Regular optimization step
                    self.encoder_optimizer.step()
                    for opt in self.decoder_optimizers:
                        opt.step()
                    
                epoch_loss += total_loss.item()
                batch_count += 1

                # Log progress
                if i % self.args.log_freq == 0:
                    current_loss = epoch_loss / (batch_count)
                    print(f"Epoch [{epoch+1}/{self.args.epochs}] Batch [{i}/{len(self.train_loader)}] Loss: {current_loss:.4f}")

            # Epoch summary
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{self.args.epochs} | Avg Loss: {avg_loss:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % self.args.save_freq == 0 or epoch == self.args.epochs - 1:
                self.save_checkpoint(epoch, self.args.output_dir, avg_loss)
            visualize_reconstruction(self,x[:8],save_path='reconstructions',epoch=epoch,args=self.args)
    

    @torch.no_grad()
    def stochastic_iterative_sampler(
        self,
        img: torch.Tensor,
        mask_ratio: float = 0.75,
        t_min=0.002,
        t_max=80.0,
        rho=7.0,
    ):  
        steps = self.T
        model = self.encoder
        
        latent, mask, ids_restore = model.forward_encoder(img, mask_ratio=mask_ratio)

        x = torch.randn_like(img) * self.sampler.sigma_max
        
        
        
        sigmas = self.sigmas
        all_steps = []
        img_patch = model.patchify(img)
        init_image = (1-mask.unsqueeze(-1)) * img_patch + mask.unsqueeze(-1) * model.patchify(x)
        init_image = model.unpatchify(init_image)
        all_steps = [init_image]
        
        for i in range(self.T):
            t = sigmas[i]
            decoder = self.decoders[i]
            # decoder = self.shared_decoder
            x0 = decoder.forward(latent,x, mask, ids_restore,t)
            x0 = model.unpatchify(x0)
            
            next_t = sigmas[i + 1]
            next_t = torch.clamp(next_t, t_min, t_max)
            delta_t = (next_t**2 - t_min**2)**0.5
            x = x0 + torch.randn_like(x) * delta_t

            
            img_patch = model.patchify(img)
            x = (1-mask.unsqueeze(-1)) * img_patch + mask.unsqueeze(-1) * model.patchify(x)
            x = model.unpatchify(x)
            all_steps.append(x)

        out = (1-mask.unsqueeze(-1)) * model.patchify(img) + mask.unsqueeze(-1) * model.patchify(x)
        out = model.unpatchify(out)
        model.train()
        return out,all_steps
    
    
    @torch.no_grad()
    def sample_heun(
        self,
        img: torch.Tensor,
        mask_ratio: float = 0.75
    ):
        """
        Heun's sampling method - a second-order variant of Euler method
        """
        model = self.encoder
        sampler = self.sampler

        sigmas = sampler.sigmas
        sigmas = sigmas.to(img.device)
        latent, mask, ids_restore = model.forward_encoder(img, mask_ratio)
        
        noise = torch.randn_like(img)
        x = noise * sampler.sigma_max 
        
        all_steps = []
        img_patch = model.patchify(img)
        init_image = (1-mask.unsqueeze(-1)) * img_patch + mask.unsqueeze(-1) * model.patchify(x)
        init_image = model.unpatchify(init_image)
        all_steps = [init_image]
        
        # Main sampling loop
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            decoder = self.decoders[i]
            # First denoising step (Euler)
            denoised = decoder.forward(latent,x, mask, ids_restore,sigma)
            denoised = model.unpatchify(denoised)
            d = (x - denoised) / sigma
            dt = sigma_next - sigma
            x_euler = x + d * dt
            
            # Second denoising step (Heun's correction)
            if sigma_next > 0:
                denoised_next = decoder.forward(latent,x_euler, mask, ids_restore,sigma_next)
                denoised_next = model.unpatchify(denoised_next)
                d_next = (x_euler - denoised_next) / sigma_next
                d_avg = (d + d_next) / 2
                x = x + d_avg * dt
            else:
                x = x_euler

            x_patch = model.patchify(img)
            x = (1-mask.unsqueeze(-1)) * x_patch + mask.unsqueeze(-1) * model.patchify(x)
            x = model.unpatchify(x)
            
            all_steps.append(x)
            

        out = (1-mask.unsqueeze(-1)) * model.patchify(img) + mask.unsqueeze(-1) * model.patchify(x)
        out = model.unpatchify(out)

        return out,all_steps




def visualize_reconstruction(model, images, save_path='reconstructions',epoch=0,args=None):
    """Visualize original, masked, and reconstructed images"""
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    print(images.shape)
    with torch.no_grad():
        # Get reconstruction and mask
        if args.vis_type == 'stochastic':
            final_image,all_steps = model.stochastic_iterative_sampler(images)
        elif args.vis_type == 'heun':
            final_image,all_steps = model.sample_heun(images)
        
        
        # Normalize images for visualization
        def normalize_image(img):
            img = img.cpu()
            # Denormalize from CIFAR-10 normalization
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0., 1.)
            return img
        
        # Normalize images for visualization
        def normalize_image(img):
            img = img.cpu()
            # Denormalize from CIFAR-10 normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0., 1.)
            return img
        
        # Prepare images for grid
        images = normalize_image(images)
        final_image = normalize_image(final_image)
        all_steps = [normalize_image(step) for step in all_steps]
        all_steps = torch.cat(all_steps,dim=0)
        # Create image grid
        n_images = min(8, images.size(0))
        comparison = torch.cat([
            images,
            final_image,
            all_steps
        ])
        
        grid = make_grid(comparison, nrow=n_images, padding=2, normalize=False)
        plt.figure(figsize=(15, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, f'reconstruction_epoch_{epoch}.png'))
        plt.close()

        return grid

def save_model(model, optimizer, scheduler, epoch, loss, save_dir='checkpoints'):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    path = os.path.join(save_dir, f'mae_epoch_{epoch}.pth')
    torch.save(checkpoint, path)
    
    # Save latest checkpoint separately
    latest_path = os.path.join(save_dir, 'mae_latest.pth')
    torch.save(checkpoint, latest_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:  # Load scaler state if it exists
            scaler = GradScaler()
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            scaler = GradScaler()
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    return 0

def get_args_parser():
    parser = argparse.ArgumentParser('MAE training for CIFAR-10', add_help=False)
    
    # Add dataset arguments
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'tiny-imagenet','imagenet-100'],
                        help='Dataset to use (cifar10 or tiny-imagenet or imagenet-100 )')
    parser.add_argument('--data_path', default='c:/dataset', type=str,
                        help='Path to dataset root directory')
    parser.add_argument('--vis_type', default='stochastic', type=str, choices=['stochastic', 'heun'],
                        help='Type of visualization')
    # Model parameters
    parser.add_argument('--model_name', default='mae_base', type=str,
                        help='Name of the model configuration')
    parser.add_argument('--img_size', default=32, type=int,
                        help='Input image size')
    parser.add_argument('--patch_size', default=4, type=int,
                        help='Patch size for image tokenization')
    parser.add_argument('--embed_dim', default=192, type=int,
                        help='Embedding dimension')
    parser.add_argument('--depth', default=12, type=int,
                        help='Depth of transformer')
    parser.add_argument('--num_heads', default=3, type=int,
                        help='Number of attention heads')
    parser.add_argument('--decoder_embed_dim', default=96, type=int,
                        help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='Depth of decoder')
    parser.add_argument('--decoder_num_heads', default=3, type=int,
                        help='Number of decoder attention heads')
    parser.add_argument('--mlp_ratio', default=4., type=float,
                        help='MLP hidden dim ratio')
    
    # Training parameters
    parser.add_argument('--epochs', default=1600, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--lr', default=1.5e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='Weight decay')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Ratio of masked patches')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Number of epochs for warmup')
    parser.add_argument('--start_warmup_value', default=1e-6, type=float,
                        help='Start warmup value')
    
    # System parameters
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='Use gradient checkpointing to save memory')
    
    # Logging and saving
    parser.add_argument('--output_dir', default='/mnt/d/repo/output/mae_cifar10_edm_wsl_lin_sample',
                        help='Path where to save checkpoints and logs')
    parser.add_argument('--save_freq', default=20, type=int,
                        help='Frequency of saving checkpoints')
    parser.add_argument('--log_freq', default=100, type=int,
                        help='Frequency of logging training progress')
    
    # Resume training
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint path')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Start epoch when resuming')
    
    # Update LR schedule arguments
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate after decay')
    parser.add_argument('--num_cycles', default=1, type=int,
                        help='Number of cycles for cosine decay')
    parser.add_argument('--warmup_lr_init', default=1e-6, type=float,
                        help='Initial learning rate for warmup')
    
    # EDML parameters
    parser.add_argument('--T', default=4, type=int,
                        help='Number of steps for EDML')
    parser.add_argument('--alpha', default=[0.01, 0.1, 0.5, 1.0], type=float, nargs='+',
                        help='Alpha values for EDML')
    # Add optimizer arguments
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=[0.5, 0.999], type=float, nargs='+',
                        help='Optimizer Betas (default: [0.9, 0.999])')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--linear_schedule', action='store_true',
                        help='Use linear schedule for alpha')
    
    return parser

def train_mae():
    args = get_args_parser().parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                              download=True, transform=transform)
    else:  # tiny-imagenet or imagenet
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load Tiny ImageNet dataset using ImageFolder
        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_path),
            transform=transform
        )

    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)
    

    model_trainer = MaskedDenoisingVit(T = args.T,args=args, device=device, trainset=trainset, train_loader=trainloader)
    

    model_trainer.train_model()

if __name__ == '__main__':
    train_mae() 