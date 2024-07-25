import torch
import math
from tqdm import tqdm
from typing import Tuple, List, Dict
import torch

class VP():
    def __init__(self,
                beta_min: float=0.1,
                beta_max: int=20,
                eps: float=1e-5,
                num_infer_step: int=1000) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps
        self.num_infer_step = num_infer_step
        
        
    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        assert len(t.shape) == 1
        return self.beta_min + (self.beta_max - self.beta_min) * t.float()
        
    def cal_interg_0_t_beta_s_ds(self, t: torch.Tensor) -> torch.Tensor:
        # linear schedule, so the integral is just the area of pic
        up = self.beta_min
        down = self.beta_t(t)
        area = (up + down) * t * 0.5
        return area
    
    def forward_drift_diffusion(self, 
                                x: torch.Tensor, 
                                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 4 and t.ndim == 1
        beta_t = self.beta_t(t)
        drift = - 0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t[:, None, None, None])
        return drift, diffusion
    
    def sample_t(self, b_s):
        # sample in range (eps, 1.0)
        return torch.rand((b_s, )) * (1 - self.eps) + self.eps
    
    def get_x_0_coefficient_and_std_t(self, t):
        interg = self.cal_interg_0_t_beta_s_ds(t)
        x_0_coefficient = torch.exp(- 0.5 * interg)
        std = torch.sqrt(1 - torch.exp(- interg))
        return x_0_coefficient[:, None, None, None], std[:, None, None, None]
    
    def prior_prob(self, shape):
        # approximately
        return torch.randn(shape)
    
    def inverse_drift_diffusion(self, x, t, pred):
        drift, diffusion = self.forward_drift_diffusion(x, t)
        # drift shape: (b, 3, h, w), diffusion shape: (b, 1, 1, 1)
        inver_drift = drift - diffusion ** 2 * pred
        inver_diffusion = diffusion
        return inver_drift, inver_diffusion
    
    @torch.no_grad()
    def sample(self, model, b_s, img_channel, sample_size, device):
        model.eval()
        # init noise
        x = self.prior_prob((b_s, img_channel, sample_size, sample_size)).to(device)
        infer_t = torch.linspace(1.0, self.eps, self.num_infer_step).to(device)
        delta_t = (1.0 - self.eps) / self.num_infer_step
        
        for i, t in enumerate(tqdm(infer_t, leave=False)):
            noise = torch.randn(x.shape, device=device)
            t = t.repeat(b_s)
            pred = model(x, t)
            inver_drift, inver_diffusion = self.inverse_drift_diffusion(x, t, pred)
            x_mean = x - inver_drift * delta_t 
            x = x_mean + inver_diffusion * math.sqrt(delta_t) * noise
        return x_mean