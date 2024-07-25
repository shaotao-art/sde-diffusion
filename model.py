from diffusers import UNet2DModel
import torch
from torch import nn
from sde import VP


class SDEDiffusion(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = UNet2DModel(**config.model_config)
        self.sche = VP(**config.sche_config)
        
    def forward(self, img, times):
        # model out -noise / std
        _, std = self.sche.get_x_0_coefficient_and_std_t(times)
        return self.model(img, times).sample / std
    
    
    def train_loss(self, batch):
        imgs = batch
        b_s = imgs.shape[0]
        times = self.sche.sample_t(b_s).to(self.config.device)
        # add noise
        x_0_coefficient, std = self.sche.get_x_0_coefficient_and_std_t(times)
        noise = torch.randn(imgs.shape, device=self.config.device)
        noisy_img = x_0_coefficient * imgs + std * noise
        # model out -noise / std * std = -noise here
        # so, in sample func, we need scale model output by div std to get logP(x) = -noise / std
        pred = self.model(noisy_img, times).sample
        # use sum in c, h, w dim, equal to scale learning rate
        train_loss = torch.mean(torch.sum((pred + noise) ** 2, dim=[1, 2, 3]))
        return train_loss
    
    @torch.no_grad()
    def sample(self, b_s, sample_size, device):
        self.model.eval()
        b_s = self.config.num_valid_gen_sample
        sample_size = self.config.model_config.sample_size
        
        imgs = self.sche.sample(self, 
                                b_s, 
                                self.config.model_config.in_channels, 
                                sample_size, 
                                device)
        return imgs
        