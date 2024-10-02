import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from utils import *
from modules import UNet

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)


class Diffusion:
    def __init__(self, noise_steps = 1000, beta_start = 1e-4, 
                 beta_end = 0.02, input_size = 256, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.input_size = input_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, 0)

    def prepare_noise_schedule(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        return betas
    
    def noise_inputs(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        E = torch.randn_like
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * E, E

    def sample_timesteps(self, n):
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))
    
    def sample(self, model, n):
        logging.info(f'Sampling {n} new samples....')
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.input_size, self.input_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), desc = 'Sampling'):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else: 
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(input_size=args.image_size, device=device)
    logger = SummaryWriter(os.path("logs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f'Epoch {epoch + 1}/{args.epochs}')
        pbar = tqdm(dataloader)
        for i, samples in enumerate(pbar):
            optimizer.zero_grad()
            x, _ = samples
            x = x.to(device)
            t = diffusion.sample_timesteps(x.size(0)).to(device)
            x_t, noise = diffusion.noise_inputs(x, t)
            predicted_noise = model(x_t, t)

            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f'Loss: {loss.item()}')
            logger.add_scalar('Loss', loss.item(), epoch * l + i)
        
        sampled_inputs = diffusion.sample(model, n=x.shape[0])
        save_images(sampled_inputs, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

    
