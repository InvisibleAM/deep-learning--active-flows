import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import sys
import pandas as pd
from tqdm import tqdm

# example
# Constants
L = Nx = Ny = 1024
DatasetSize = 4
NumParticles = 4
GlobalParticleSize = 8
minParticleSeparation = 6 * GlobalParticleSize

# Function to generate positions
import math
def generate_dots_array2D(num_dots, gridSize=128):
    #dots_array = []
    x_array = []
    y_array = []

    margin = 20
    grid_dim = math.ceil(math.sqrt(num_dots))  # Determine how many rows/cols to split into
    subgrid_size = gridSize // grid_dim

    subgrid_indices = [(i, j) for i in range(grid_dim) for j in range(grid_dim)]
    #np.random.shuffle(subgrid_indices)  # Shuffle to randomize which subgrids are used

    if num_dots > len(subgrid_indices):
        raise ValueError("Cannot place more dots than subgrids.")

    for idx in range(num_dots):
        i, j = subgrid_indices[idx]
        origin_x = i * subgrid_size
        origin_y = j * subgrid_size

        # while True:
        x = np.random.randint(origin_x + margin, origin_x + subgrid_size - margin)
        y = np.random.randint(origin_y + margin, origin_y + subgrid_size - margin)
        #if all(np.linalg.norm(np.array([x, y]) - np.array(dot)) >= min_distance for dot in dots_array):
        x_array.append(x)
        y_array.append(y)
        #dots_array.append([x, y])

    return np.array(x_array + y_array)

# Function to calculate velocity and force fields
def vCalculatortot(r, p, Np, particleSize, f0, d0, Nx=1024, Ny=1024):
    x, y = np.meshgrid(range(Nx), range(Ny))

    Fxk = np.zeros((Nx, Ny), dtype=np.complex128)
    Fyk = np.zeros((Nx, Ny), dtype=np.complex128)

    Fxk2s = np.zeros((Nx, Ny), dtype=np.complex128)
    Fyk2s = np.zeros((Nx, Ny), dtype=np.complex128)

    Fxk3t = np.zeros((Nx, Ny), dtype=np.complex128)
    Fyk3t = np.zeros((Nx, Ny), dtype=np.complex128)

    # Fourier grid
    kx_ = (2 * np.pi / 1) * np.fft.fftfreq(Nx)
    ky_ = (2 * np.pi / 1) * np.fft.fftfreq(Ny)
    kx, ky = np.meshgrid(kx_, ky_)

    sigma = particleSize * np.sqrt(2 / np.pi)
    scale = 1 / (2 * np.pi * sigma**2)

    arg = ((x - Nx / 2)**2 + (y - Ny / 2)**2) / (2 * sigma**2)
    fx0 = np.exp(-arg) * scale
    Fk0 = np.fft.fft2(fx0)

    k2 = kx * kx + ky * ky
    ik2 = 1 / k2
    ik2[ik2 > 1e6] = 0

    # Calculations in Fourier space
    for i in range(Np):
        pdotk = p[i, 0] * kx + p[i, 1] * ky
        kdotr = kx * (r[i, 0] - Nx / 2) + ky * (r[i, 1] - Ny / 2)
        orientation_vector = 0.01* particleSize * p[i]
        kdotrdipole_positve = kx * (r[i, 0] + orientation_vector[0] - Nx / 2) + ky * (r[i, 1] + orientation_vector[1] - Ny / 2)
        kdotrdipole_negative = kx * (r[i, 0] - orientation_vector[0] - Nx / 2) + ky * (r[i, 1] - orientation_vector[1] - Ny / 2)

        Fxk2s += -Fk0 * np.exp(-1j * kdotrdipole_positve) * p[i,0] * f0[i]
        Fyk2s += -Fk0 * np.exp(-1j * kdotrdipole_positve) * p[i,1] * f0[i]

        Fxk2s += -Fk0 * np.exp(-1j * kdotrdipole_negative) * p[i,0] * (-f0[i])
        Fyk2s += -Fk0 * np.exp(-1j * kdotrdipole_negative) * p[i,1] * (-f0[i])

        Fxk3t += -Fk0 * np.exp(-1j * kdotr) * p[i, 0] * k2 * d0[i]
        Fyk3t += -Fk0 * np.exp(-1j * kdotr) * p[i, 1] * k2 * d0[i]

    fx2s = np.real(np.fft.ifft2(Fxk2s))
    fy2s = np.real(np.fft.ifft2(Fyk2s))

    fx3t = np.real(np.fft.ifft2(Fxk3t))
    fy3t = np.real(np.fft.ifft2(Fyk3t))

    fx = fx2s + fx3t
    fy = fy2s + fy3t


    Fdotk2s = Fxk2s * kx + Fyk2s * ky
    vxk2s = (Fxk2s - Fdotk2s * (kx * ik2)) * ik2
    vyk2s = (Fyk2s - Fdotk2s * (ky * ik2)) * ik2
    vxk2s[0, 0] = 0
    vyk2s[0, 0] = 0

    Fdotk3t = Fxk3t * kx + Fyk3t * ky
    vxk3t = (Fxk3t - Fdotk3t * (kx * ik2)) * ik2
    vyk3t = (Fyk3t - Fdotk3t * (ky * ik2)) * ik2
    vxk3t[0, 0] = 0
    vyk3t[0, 0] = 0

    vx2s = np.real(np.fft.ifft2(vxk2s))
    vy2s = np.real(np.fft.ifft2(vyk2s))

    vx3t = np.real(np.fft.ifft2(vxk3t))
    vy3t = np.real(np.fft.ifft2(vyk3t))

    vx = vx2s + vx3t
    vy = vy2s + vy3t

    return vx, vy, fx, fy


# Initialize arrays to store data for each dataset
posArray = np.zeros((DatasetSize, NumParticles, 2))
Orientation = np.zeros((DatasetSize, NumParticles, 2))
f0 = np.zeros((DatasetSize, NumParticles))
d0 = np.zeros((DatasetSize, NumParticles))
vx = np.zeros((DatasetSize, Nx, Ny))
vy = np.zeros((DatasetSize, Nx, Ny))
F_x = np.zeros((DatasetSize, Nx, Ny))
F_y = np.zeros((DatasetSize, Nx, Ny))

vx_n = np.zeros((DatasetSize, 128, 128))
vy_n = np.zeros((DatasetSize, 128, 128))
F_x_n = np.zeros((DatasetSize, 128, 128))
F_y_n = np.zeros((DatasetSize, 128, 128))




thetalist = np.radians(np.arange(0, 360, 2))  # Convert angles from degrees to radians

# Generate data for each dataset
for i in tqdm(range(DatasetSize), desc="Generating dataset"):
    generated_positions = generate_dots_array2D(NumParticles)
    x_positions = (1024 // 2 - 64) + generated_positions[:NumParticles]
    y_positions = (1024 // 2 - 64) + generated_positions[NumParticles:]
    posArray[i] = np.column_stack((x_positions, y_positions))

    ori_theta = np.random.choice(thetalist, NumParticles)

    Orientation[i] = np.column_stack((np.cos(ori_theta), np.sin(ori_theta)))
    f0[i] = np.random.uniform(40000, 60000, NumParticles)
    d0[i] = np.random.uniform(12000, 16000, NumParticles)
    vx[i], vy[i], F_x[i], F_y[i] = vCalculatortot(
        posArray[i], Orientation[i], NumParticles, GlobalParticleSize, f0[i], d0[i], Nx=1024, Ny=1024
    )
    minipos = 1024//2 - 64
    maxipos = 1024//2 + 64
    vx_n[i] = vx[i][minipos:maxipos, minipos:maxipos]
    vy_n[i] = vy[i][minipos:maxipos, minipos:maxipos]
    F_x_n[i] = F_x[i][minipos:maxipos, minipos:maxipos]
    F_y_n[i] = F_y[i][minipos:maxipos, minipos:maxipos]

data = []

for i in range(DatasetSize):
    data.append({
        'Position': posArray[i],  # Shape: NumParticles x 2
        'Orientation': Orientation[i],  # Shape: NumParticles x 2
        'f0' :f0[i],  # Shape: NumParticles
        'd0': d0[i],  # Shape: NumParticles
        'vx': vx_n[i],  # Shape: Nx x Ny
        'vy': vy_n[i],  # Shape: Nx x Ny
        'fx': F_x_n[i],  # Shape: Nx x Ny
        'fy': F_y_n[i]   # Shape: Nx x Ny
    })

df = pd.DataFrame(data)


# Shuffle the DataFrame
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split sizes
total_size = len(shuffled_df)
train_size = int(0.7 * total_size)
test_size = int(0.15 * total_size)
eval_size = total_size - train_size - test_size

# Split the DataFrame
train_df = shuffled_df[:train_size]
test_df = shuffled_df[train_size:train_size+test_size]
eval_df = shuffled_df[train_size+test_size:]

# Save the train, eval, and test DataFrames as .npz files
np.savez('train_df_128_4p.npz', **{col: train_df[col].to_numpy() for col in train_df.columns})
np.savez('eval_df_128_4p.npz', **{col: eval_df[col].to_numpy() for col in eval_df.columns})
np.savez('test_df_128_4p.npz', **{col: test_df[col].to_numpy() for col in test_df.columns})