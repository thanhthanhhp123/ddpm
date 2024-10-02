import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from torchvision import datasets, transforms    

from utils import *
from modules import UNet
from ddpm import Diffusion

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transforms_ = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
])

train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transforms_)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)

test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transforms_)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)

model = UNet(in_channels = 1, out_channels = 1)
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr = 1e-4)
mse = nn.MSELoss()
diffusion = Diffusion(input_size=28, device=device)

for epoch in range(100):
    model.train()
    pbar = tqdm(train_loader, desc = f'Epoch {epoch}')
    for x, _ in pbar:
        x = x.to(device)
        t = diffusion.sample_timesteps(x.size(0)).to(device)
        x_t, noise = diffusion.noise_inputs(x, t)
        predicted_noise = model(x_t, t)

        loss = mse(predicted_noise, noise)

        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar.set_postfix({'Loss': loss.item()})

    sampled_inputs = diffusion.sample(model, n = x.shape[0])
    save_images(sampled_inputs, f"results/{epoch}.jpg")
    torch.save(model.state_dict(), f"models/ckpt.pt")