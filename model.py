import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.fft
import torch.nn.functional as F

import os 
import sys
import time
import h5py
from tqdm import tqdm

class V2FCNN2v2(nn.Module):
    def __init__(self):
        super(V2FCNN2v2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # Input channels: 2 (vx, vy)
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),


            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(16, 2, kernel_size=3, padding=1)  # Output channels: 2
        )

    def forward(self, x):
        x = self.net(x)
        return x
