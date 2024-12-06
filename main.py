import torch

from model.DDPM import UNet
\
from model.DDPM import ddpm_inpainting_loss, NoiseScheduler
from datasets.dataset import ArtDataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':    
    use_gpu = torch.cuda.is_available()
    device = torch.device('cpu')
    if use_gpu:
        print("CUDA is available.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")


    config = {
        'batchsize':8,

    }
    batch_size, channels, height, width = 8, 3, 64, 64
    x_0 = torch.rand(batch_size, channels, height, width).cuda()  # 원본 이미지
    mask = torch.randint(0, 2, (batch_size, 1, height, width)).float().cuda()  # 마스크 (손상된 영역)
    timesteps = torch.randint(0, 1000, (batch_size,)).cuda()  # 랜덤 타임스텝

    # 모델 및 Noise Scheduler 초기화
    model = UNet().cuda()
    noise_scheduler = NoiseScheduler(beta_start=0.1, beta_end=0.2, timesteps=1000, device=device)

    # 손실 계산
    loss = ddpm_inpainting_loss(model, x_0, mask, timesteps, noise_scheduler)
    print(f"Loss: {loss.item()}")