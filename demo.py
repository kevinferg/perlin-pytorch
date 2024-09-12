import numpy as np
import matplotlib.pyplot as plt
import torch
from noise import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_noise_demo(noise_model, imgname, res=4, periodic=False, smoothness=2, n_dims=2, n_fields=8, seed=0, n_points = 10000):

    model = noise_model(n_dims=n_dims, n_fields=n_fields, res=res, seed=seed, periodic=periodic, smoothness=smoothness)
    x = torch.rand(n_points, n_dims)
    fields = model(x)
    
    plt.figure(figsize=(n_fields*2,2), dpi=200)
    for i in range(n_fields):
        plt.subplot(1, n_fields, i+1)
        plt.scatter(x[:,0].detach().numpy(), x[:,1].detach().numpy(), s=4,c=fields[:,i].detach().numpy(), cmap='jet')
        plt.axis("equal")
        plt.axis("off")
    plt.suptitle(f"{model.n_dims}D {model.name}:    Resolution = {model.res},   Smoothness = {model.smoothness},   Periodic = {model.periodic}")
    plt.savefig(imgname, bbox_inches="tight")

if __name__ == "__main__":
    plot_noise_demo( ValueNoise,  "value_demo.png", res=4, periodic=True, smoothness=1)
    plot_noise_demo(PerlinNoise, "perlin_demo.png", res=2, periodic=False, smoothness=3)