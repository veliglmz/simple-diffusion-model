import numpy as np
import torch


def generate_spiral_data(N):
    u = torch.rand(N)
    rho = u * 0.65 + 0.25 + torch.rand(N) * 0.15
    theta = u * np.pi * 3
    data = torch.empty(N, 2)
    data[:, 0] = theta.cos() * rho
    data[:, 1] = theta.sin() * rho
    return data