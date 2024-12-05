import torch

from model.DDPM import UNet
from model.noisescheduler import NoiseScheduler

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


    model = UNet().to(device)
    noise_scheduler = NoiseScheduler(timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    #----------------------구현필요--------------------------------------------