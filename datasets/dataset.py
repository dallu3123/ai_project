import torch
import os
import random 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

class ArtDataset(Dataset):
    def __init__(self, image_dir, image_size=128, mask_size_range=(16,64), transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', ',jpeg'))]
        self.image_size = image_size
        self.mask_size_range = mask_size_range
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        #decide mask size (마스크 사이즈 결정)
        mask_size = random.randint(*self.mask_size_range)
        mask = torch.ones_like(image)
        x = random.randint(0, self.image_size - mask_size)
        y = random.randint(0, self.image_size - mask_size)
        mask[:, y:y+mask_size, x:x+mask_size] = 0  # mask area(마스크 영역)

        masked_image = image * mask
        return masked_image, image, mask
