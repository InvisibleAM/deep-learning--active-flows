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

import dill

from model import *

train_data = np.load('/home/invisibleam/ml-fluidDynamics/Stokes_flow_DL_higres_4p/train_df_128_4p_nonperiodic.npz', allow_pickle=True)
test_data = np.load('/home/invisibleam/ml-fluidDynamics/Stokes_flow_DL_higres_4p/test_df_128_4p_nonperiodic.npz', allow_pickle=True)
eval_data = np.load('/home/invisibleam/ml-fluidDynamics/Stokes_flow_DL_higres_4p/eval_df_128_4p_nonperiodic.npz', allow_pickle=True)

# Convert the loaded data back to DataFrames
train_df = pd.DataFrame({key: train_data[key] for key in train_data.files})
test_df = pd.DataFrame({key: test_data[key] for key in test_data.files})
eval_df = pd.DataFrame({key: eval_data[key] for key in eval_data.files})

print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Eval set size: {len(eval_df)}")

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = torch.tensor([row['vx'], row['vy']], dtype=torch.float32)
        y = torch.tensor([row['fx'], row['fy']], dtype=torch.float32)
        return x, y

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(CustomDataset(train_df), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CustomDataset(test_df), batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(CustomDataset(eval_df), batch_size=batch_size, shuffle=True)

# Example usage
for batch_x, batch_y in train_loader:
    print("Input batch shape:", batch_x.shape)
    print("Output batch shape:", batch_y.shape)
    break


# Initialize the model, loss function and optimizer

model = V2FCNN2v2()  # Create an instance of the model
#sigma_loss = 0.5*particleSize * np.sqrt(2 / np.pi)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(batch_x)  # Forward pass
        loss = criterion(outputs, batch_y)  # Calculate loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluation loop on the eval set
    model.eval()  # Set the model to evaluation mode
    eval_loss = 0.0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch_x, batch_y in eval_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            eval_loss += loss.item()

    eval_loss /= len(eval_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.10f}, Eval Loss: {eval_loss:.10f}')

# Testing loop on the test set
model.eval()  # Set the model to evaluation mode for testing
test_loss = 0.0

with torch.no_grad():  # No need to compute gradients during testing
    for batch_x, batch_y in tqdm(test_loader, desc="Testing", leave=False):
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.10f}')

torch.save(model, 'model_non_periodic.pth')