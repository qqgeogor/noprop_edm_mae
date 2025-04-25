import torch
import numpy as np
from typing import Callable, Optional, Tuple
import math


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


class KarrasSampler:
    """
    Implementation of Karras et al. EDM sampler with noise schedule
    """
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_steps: int = 40,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float('inf'),
        S_noise: float = 1.0,
        sampling: str = 'log',
    ):
        """
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            rho: Controls the sampling schedule
            num_steps: Number of sampling steps
            S_churn: Stochasticity strength (0 = deterministic)
            S_min: Minimum sigma for stochastic sampling
            S_max: Maximum sigma for stochastic sampling
            S_noise: Noise level for stochastic sampling
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.sigmas = self.get_sigma_schedule()
        self.sampling = sampling

    def get_sigma_schedule(self) -> torch.Tensor:
        """
        Generate noise schedule based on Karras et al.
        Returns:
            Tensor of sigma values for each step
        """
        steps = torch.arange(self.num_steps + 1, dtype=torch.float32)
        t = steps / self.num_steps
        inv_rho = 1.0 / self.rho
        
        # Compute sigmas according to EDM paper
        sigma = self.sigma_min ** (1/self.rho) + t * (
            self.sigma_max ** (1/self.rho) - self.sigma_min ** (1/self.rho)
        )
        sigmas = sigma ** self.rho
        return sigmas

    def get_subsequent_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Get next sigma value in the sequence
        """
        # Find current sigma index
        sigma = sigma.detach().cpu().numpy()
        sigmas = self.sigmas.cpu().numpy()
        
        index = np.searchsorted(sigmas, sigma)
        if index == 0:
            return torch.tensor(sigmas[0], device=sigma.device)
        return torch.tensor(sigmas[index - 1], device=sigma.device)

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        x: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        extra_args: dict = {},
        callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Perform sampling using Euler method with stochastic sampling
        
        Args:
            model: Model that predicts denoised sample or noise
            x: Initial noise tensor
            sigmas: Optional custom sigma schedule
            extra_args: Extra arguments to pass to model
            callback: Optional callback function for each step
            
        Returns:
            Denoised sample
        """
        # Get sigma schedule if not provided
        if sigmas is None:
            sigmas = self.sigmas
        
        # Move sigmas to same device as x
        sigmas = sigmas.to(x.device)
        
        # Main sampling loop
        for i in range(len(sigmas) - 1):
            gamma = min(self.S_churn / self.num_steps, 2 ** 0.5 - 1) if self.S_min <= sigmas[i] <= self.S_max else 0
            
            # Add noise if using stochastic sampling
            eps = torch.randn_like(x) * self.S_noise if gamma > 0 else 0
            sigma_hat = sigmas[i] * (gamma + 1)
            
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            
            # Euler step
            denoised = model(x, sigma_hat, **extra_args)
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            
            # Update x
            x = x + d * dt
            
            # Apply callback if provided
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat})
        
        return x

    @torch.no_grad()
    def sample_euler_ancestral(
        self,
        model: Callable,
        img: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        extra_args: dict = {},
        callback: Optional[Callable] = None,
        mask_ratio: float = 0.75
    ) -> torch.Tensor:
        """
        Ancestral sampling with Euler method
        """
        if sigmas is None:
            sigmas = self.sigmas
        sigmas = sigmas.to(img.device)
        latent, mask, ids_restore = model.forward_encoder(img,mask_ratio)
        
        noise = torch.randn_like(img)
        x = noise * self.sigma_max 
        # Main sampling loop
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]

            # Euler step
            denoised = model.denoise(x, latent, mask,ids_restore,sigma)
            d = (x - denoised) / sigma
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
            
            #Add noise
            if sigmas[i + 1] > 0:
                noise = torch.randn_like(x)
                x = x + noise * sigmas[i + 1]
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigma})

        # print(x.shape,mask.shape,img.shape, model.patchify(img).shape, model.patchify(x).shape)

        out = (1-mask.unsqueeze(-1)) * model.patchify(img) + mask.unsqueeze(-1) * model.patchify(x)
        out = model.unpatchify(out)

        return out,mask
    

    def sample_euler(
        self,
        model: Callable,
        img: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        extra_args: dict = {},
        callback: Optional[Callable] = None,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Ancestral sampling with Euler method
        """
        if sigmas is None:
            sigmas = self.sigmas
        sigmas = sigmas.to(img.device)
        latent, mask, ids_restore = model.forward_encoder(img,mask_ratio)
        
        noise = torch.randn_like(img)
        x = noise * self.sigma_max 
        # Main sampling loop
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]

            # Euler step
            denoised = model.denoise(x, latent, mask,ids_restore,sigma)
            d = (x - denoised) / sigma
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
            
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigma})

        out = (1-mask.unsqueeze(-1)) * model.patchify(img) + mask.unsqueeze(-1) * model.patchify(x)
        out = model.unpatchify(out)

        return out,mask

    @torch.no_grad()
    def sample_euler_unet(
        self,
        model: Callable,
        img: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        extra_args: dict = {},
        callback: Optional[Callable] = None,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Sampling with Euler method
        """
        if sigmas is None:
            sigmas = self.sigmas
        sigmas = sigmas.to(img.device).view(-1,1,1,1)
        
        noise = torch.randn_like(img)
        x = noise * self.sigma_max 
        # Main sampling loop
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]

            # Euler step
            denoised,_,_,_ = model(x,x)
            d = (x - denoised) / sigma
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
            
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigma})
        return x
    
    @torch.no_grad()
    def stochastic_iterative_sampler_unet(
        self,
        model: Callable,
        img: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        extra_args: dict = {},
        callback: Optional[Callable] = None,
        mask_ratio: float = 0.75,
        t_min=0.002,
        t_max=80.0,
        rho=7.0,
        steps=40,
    ):  
        model.eval()
        if sigmas is None:
            sigmas = self.sigmas
        sigmas = sigmas.to(img.device).view(-1,1,1,1)
        
        noise = torch.randn_like(img)
        x = noise * self.sigma_max 

        
        ts = torch.linspace(0,steps-1,20)
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        
        for i in range(len(ts) - 1):
            t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
            x0,_,_,_ = model(x,x)
            next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
            next_t = np.clip(next_t, t_min, t_max)
            x = x0 + torch.randn_like(x) * (next_t**2 - t_min**2)**0.5

        model.train()
        return x
    
    @torch.no_grad()
    def stochastic_iterative_sampler(
        self,
        model: Callable,
        img: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        extra_args: dict = {},
        callback: Optional[Callable] = None,
        mask_ratio: float = 0.75,
        t_min=0.002,
        t_max=80.0,
        rho=7.0,
        steps=40,
    ):  
        
        model.eval()
        latent, mask, ids_restore = model.forward_encoder(img, mask_ratio=mask_ratio)

        x = torch.randn_like(img) * self.sigma_max

        
        ts = torch.linspace(0,steps-1,20)
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)

        for i in range(len(ts) - 1):
            t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
            x0 = model.denoise(x, latent, mask, ids_restore,t)
            next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
            next_t = np.clip(next_t, t_min, t_max)
            x = x0 + torch.randn_like(x) * (next_t**2 - t_min**2)**0.5

        out = (1-mask.unsqueeze(-1)) * model.patchify(img) + mask.unsqueeze(-1) * model.patchify(x)
        out = model.unpatchify(out)
        model.train()
        return out,mask

    @torch.no_grad()
    def sample_heun(
        self,
        model: Callable,
        img: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        extra_args: dict = {},
        callback: Optional[Callable] = None,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Heun's sampling method - a second-order variant of Euler method
        """
        if sigmas is None:
            sigmas = self.sigmas
        sigmas = sigmas.to(img.device)
        latent, mask, ids_restore = model.forward_encoder(img, mask_ratio)
        
        noise = torch.randn_like(img)
        x = noise * self.sigma_max 
        
        # Main sampling loop
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # First denoising step (Euler)
            denoised = model.denoise(x, latent, mask, ids_restore,sigma)
            d = (x - denoised) / sigma
            dt = sigma_next - sigma
            x_euler = x + d * dt
            
            # Second denoising step (Heun's correction)
            if sigma_next > 0:
                denoised_next = model.denoise(x_euler, latent, mask, ids_restore,sigma_next)
                d_next = (x_euler - denoised_next) / sigma_next
                d_avg = (d + d_next) / 2
                x = x + d_avg * dt
            else:
                x = x_euler
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigma})

        out = (1-mask.unsqueeze(-1)) * model.patchify(img) + mask.unsqueeze(-1) * model.patchify(x)
        out = model.unpatchify(out)

        return out,mask

    @torch.no_grad()
    def sample_euler_single_class(
        self,
        model: Callable,
        img: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        extra_args: dict = {},
        callback: Optional[Callable] = None,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Ancestral sampling with Euler method for single class
        """
        if sigmas is None:
            sigmas = self.sigmas
        sigmas = sigmas.to(img.device)
        latent, mask, ids_restore = model.forward_encoder(img,mask_ratio)
        
        noise = torch.randn_like(img)
        x = noise * self.sigma_max 
        # Main sampling loop
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]

            # Euler step
            denoised = model.denoise(x, latent, mask,ids_restore,sigma)
            d = (x - denoised) / sigma
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
            
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigma})

        
        return x,mask

    def add_noise(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to image according to EDM paper
        
        Args:
            x: Clean image tensor [B, C, H, W]
            noise: Optional pre-generated noise (if None, will generate random noise)
            sigma: Optional noise level (if None, will sample randomly from schedule)
        
        Returns:
            Tuple of (noisy_image, noise_level)
        """
        # Generate random noise if not provided
        if noise is None:
            noise = torch.randn_like(x)
        
        # Sample random sigma if not provided
        if sigma is None:
            # Generate random uniform values between 0 and 1
            # indices = torch.randint(
            #     0, self.num_steps - 1, (x.shape[0],), device=x.device
            # )
            # sigma = sigmas[indices].view(-1,1,1,1)
            sigma = self.get_random_sigma(torch.rand(x.shape[0]),sampling=self.sampling).to(x.device)
        else:
            sigma = sigma.view(-1,1,1,1)
        # Add noise to image
        noisy = x + noise * sigma
        
        return noisy, sigma


    def get_random_sigma(self, u: torch.Tensor, sampling='log') -> torch.Tensor:
        if sampling == 'log':
            # Log-uniform sampling
            log_sigma_min = math.log(self.sigma_min)
            log_sigma_max = math.log(self.sigma_max)
            log_sigma = log_sigma_min + u * (log_sigma_max - log_sigma_min)
            sigma = torch.exp(log_sigma)
        else:
            # Uniform sampling
            sigma = self.sigma_min + u * (self.sigma_max - self.sigma_min)
        
        return sigma.view(-1,1,1,1)

def get_ancestral_step(sigma_from: torch.Tensor, sigma_to: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate sigma and noise magnitude for ancestral sampling step
    """
    sigma_up = torch.minimum(sigma_to, (sigma_from ** 2 * (sigma_to ** 2 - sigma_from ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

class EDMPrecond:
    """
    EDM preconditioner for model outputs
    """
    def __init__(self, inner_model: Callable):
        self.inner_model = inner_model

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        c_skip = sigma ** 2 / (sigma ** 2 + 1)
        c_out = 1 / (sigma ** 2 + 1) ** 0.5
        c_in = 1 / (sigma ** 2 + 1) ** 0.5
        c_noise = 0.25 * sigma.log()
        
        x_in = c_in * x
        sigma_in = c_noise * torch.ones_like(sigma)
        
        F_x = self.inner_model(x_in, sigma_in, **kwargs)
        D_x = c_skip * x + c_out * F_x
        
        return D_x
    



class EDMPrecondMae:
    """
    EDM preconditioner for model outputs
    """
    def __init__(self, inner_model: Callable):
        self.inner_model = inner_model
        self.sigma_data = 0.5

    def __call__(self, latent, noised_image, mask, ids_restore,sigma, **kwargs) -> torch.Tensor:
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        x_in = c_in * noised_image
        
        F_x = self.inner_model.forward_decoder(latent, x_in, mask, ids_restore)
        F_x = self.inner_model.unpatchify(F_x)
        
        D_x = c_skip * noised_image + c_out * F_x
        
        D_x = self.inner_model.patchify(D_x)
        
        return D_x

def sample_example():
    """Example usage of the sampler"""
    # Define a dummy model
    def dummy_model(x, sigma, **kwargs):
        return x  # Replace with actual model
    
    # Initialize sampler
    sampler = KarrasSampler(
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        num_steps=40
    )
    
    # Create random noise
    x = torch.randn(1, 3, 64, 64)  # [B, C, H, W]
    
    # Sample using Euler method
    result = sampler.sample(dummy_model, x)
    
    # Sample using ancestral sampling
    result_ancestral = sampler.sample_euler_ancestral(dummy_model, x)
    
    return result, result_ancestral

def noise_example():
    """Example of adding noise to images"""
    # Initialize sampler
    sampler = KarrasSampler(
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        num_steps=40
    )
    
    # Create dummy image (or load real image)
    clean_image = torch.randn(4, 3, 32, 32)  # [B, C, H, W]
    
    # Add noise with random sigma
    noisy_image, sigma = sampler.add_noise(clean_image)
    print(f"Added noise with sigma: {sigma}")
    
    # Add noise with specific sigma
    specific_sigma = torch.tensor([0.5, 1.0, 2.0, 4.0], device=clean_image.device)
    noisy_image_fixed, _ = sampler.add_noise(clean_image, sigma=specific_sigma)
    
    # Add specific noise
    custom_noise = torch.randn_like(clean_image)
    noisy_image_custom, sigma = sampler.add_noise(clean_image, noise=custom_noise)
    
    return noisy_image, noisy_image_fixed, noisy_image_custom

if __name__ == "__main__":
    # Run example
    result, result_ancestral = sample_example()
    noisy1, noisy2, noisy3 = noise_example()
