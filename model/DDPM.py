import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, downsample=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels), 
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.downsample = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1) if downsample else None
    
    def forward(self, x, t):
        time_emb = self.time_mlp(t).view(t.shape[0], -1, 1, 1)
        x=self.conv1(x) + time_emb
        x=F.relu(x)
        x=self.conv2(x)
        if self.downsample:
            x = self.downsample(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, time_emb_dim=256):
        super().__init__()
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        #입력 채널 + mask이미지 = 3 + 1 = 4
        self.block1 = Block(4,64, time_emb_dim)
        self.block2 = Block(64, 128, time_emb_dim, downsample=True)
        self.block3 = Block(128, 256, time_emb_dim, downsample=True)

        self.middle_block=Block(256, 256, time_emb_dim)
        self.block4 = Block(256, 128, time_emb_dim)
        self.block5 = Block(128, 64, time_emb_dim)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, mask, t):
        t_emb = self.time_embedding(t)
        #입력 채널 확장: 이미지 + 마스크 
        x=torch.cat([x, mask], dim=1)
    
        x1 = self.block1(x, t_emb)
        x2 = self.block2(x1, t_emb)
        x3 = self.block3(x2, t_emb)

        x_middle = self.middle_block(x3, t_emb)

        x4 = F.interpolate(self.block4(x_middle, t_emb), scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.interpolate(self.block5(x4, t_emb), scale_factor=2, mode="bilinear", align_corners=False)

        return self.final_conv(x5)

def ddpm_loss(model, x_0, timesteps, noise_scheduler):
    noise = torch.randn_like(x_0)
    x_t = noise_scheduler.add_noise(x_0, noise, timesteps)
    predicted_noise = model(x_t, timesteps)
    return F.mse_loss(predicted_noise, noise)

def ddpm_inpainting_loss(model, x_0, mask, timesteps, noise_scheduler):
    """
    #손상된 영역만 손실 계산
    #Inpainting_loss function
    """
    noise = torch.rand_like(x_0)
    x_t = noise_scheduler.add_noise(x_0, noise, timesteps)
    masked_x_t = x_t * mask + x_0 * (1 - mask)
    predicted_noise = model(masked_x_t, mask, timesteps)

    return F.mse_loss(predicted_noise * mask, noise * mask)

class NoiseScheduler:
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0) #dim 차원에 있는 input 요소의 누적 곱을 반환
        self.device = device

    def add_noise(self, x_0, noise, t):
        t = t.to(self.alpha_cumprod.device)
        print(t.device)
        sqrt_alpha_cumprod = self.alpha_cumprod[t] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.to(self.device)
        sqrt_one_minus_alpha_cumprod = (1 - self.alpha_cumprod[t]) ** 0.5
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.to(self.device)
        print(sqrt_one_minus_alpha_cumprod.device, x_0.device)
        return sqrt_alpha_cumprod[:, None, None, None] * x_0 + \
               sqrt_one_minus_alpha_cumprod[:, None, None, None] * noise

