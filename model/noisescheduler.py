import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from DDPM import UNet

class NoiseScheduler:
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0) #dim 차원에 있는 input 요소의 누적 곱을 반환

    def add_noise(self, x_0, noise, t):
        sqrt_alpha_cumprod = self.alpha_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_cumprod = (1 - self.alpha_cumprod[t]) ** 0.5
        return sqrt_alpha_cumprod[:, None, None, None] * x_0 + \
               sqrt_one_minus_alpha_cumprod[:, None, None, None] * noise


        