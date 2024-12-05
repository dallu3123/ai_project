import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np

class InpaintingDataset(Dataset):
    def __init__(self):
        super.__init__()