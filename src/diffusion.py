'''
Module to implement the diffusion process
'''

# Import the required libraries
from time import perf_counter
import torch
from torch.nn import functional as F

# Define the function to add noise to the clean signal
def add_noise(clean, noise, timesteps, alpha_bars):
    # Look up cumulative signal retention for each example
    alpha_bar = alpha_bars[timesteps]
    broadcast_shape = (-1,) + (1,) * (clean.ndim - 1)
    alpha_bar = alpha_bar.reshape(broadcast_shape)

    # Combine the retained signal with Gaussian noise
    return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

# Define the function to compute the noise prediction loss
def noise_prediction_loss(denoiser, clean, alpha_bars, rng = None):
    # Draw one diffusion step per example
    timesteps = torch.randint(
        len(alpha_bars), (len(clean),), device = clean.device, generator = rng
    )

    # Draw the noise target and create the noisy training inputs
    noise = torch.randn(
        clean.shape, dtype = clean.dtype, device = clean.device, generator = rng
    )
    noisy = add_noise(clean, noise, timesteps, alpha_bars)

    # Compare the predicted noise with the noise that was added
    return F.mse_loss(denoiser(noisy, timesteps), noise)

# Define the function to train the diffusion model
def train_diffusion(
    denoiser, data, betas, seed, *, steps, batch_size, lr, use_cosine = False
):
    # Prepare the noise schedule, optimizer, and reproducible batch sampling
    alpha_bars = (1 - betas).cumprod(dim = 0)
    rng = torch.Generator(device = data.device).manual_seed(seed)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr = lr)
    scheduler = None

    # Enable cosine learning-rate decay for the digits experiment
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = steps)

    # Record the loss at every training step
    denoiser.train()
    loss_history = []
    start_time = perf_counter()

    for _ in range(steps):
        # Draw a batch and update the denoiser
        indices = torch.randint(
            len(data), (batch_size,), device = data.device, generator = rng
        )
        optimizer.zero_grad()
        loss = noise_prediction_loss(denoiser, data[indices], alpha_bars, rng = rng)
        loss.backward()
        optimizer.step()

        # Update the learning rate after the optimizer step
        if scheduler is not None:
            scheduler.step()

        loss_history.append(loss.item())

    # Return the loss history and elapsed training time
    return loss_history, perf_counter() - start_time

# Define the function to sample the ancestral process
@torch.no_grad()
def sample_ancestral(denoiser, betas, initial_noise, rng = None):
    # Compute per-step and cumulative signal retention
    alphas = 1 - betas
    alpha_bars = alphas.cumprod(dim = 0)
    samples = initial_noise

    for index in reversed(range(len(betas))):
        # Predict the noise remaining at the current reverse step
        timesteps = torch.full(
            (len(samples),), index, dtype = torch.long, device = samples.device
        )
        predicted_noise = denoiser(samples, timesteps)

        # Compute the mean of the reverse update
        noise_scale = betas[index] / (1 - alpha_bars[index]).sqrt()
        samples = (samples - noise_scale * predicted_noise) / alphas[index].sqrt()

        if index > 0:
            # Add noise using the posterior variance before the final step
            variance = betas[index] * (1 - alpha_bars[index - 1]) / (1 - alpha_bars[index])
            step_noise = torch.randn(
                samples.shape,
                dtype = samples.dtype,
                device = samples.device,
                generator = rng,
            )
            samples = samples + variance.sqrt() * step_noise

    return samples

# Define the function to sample the deterministic process
@torch.no_grad()
def sample_deterministic(denoiser, betas, initial_noise, steps):
    # Select evenly spaced steps in descending order
    alpha_bars = (1 - betas).cumprod(dim = 0)
    indices = torch.linspace(len(betas) - 1, 0, steps = steps).round().long().tolist()
    samples = initial_noise

    for position, index in enumerate(indices):
        # Predict the noise at the current step
        timesteps = torch.full(
            (len(samples),), index, dtype = torch.long, device = samples.device
        )
        predicted_noise = denoiser(samples, timesteps)
        alpha_bar = alpha_bars[index]

        # Estimate the clean sample from the current noisy sample
        clean = (samples - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()

        # Return the clean estimate after the final model evaluation
        if position == len(indices) - 1:
            return clean

        # Move to the next selected step using the same predicted noise
        next_alpha_bar = alpha_bars[indices[position + 1]]
        samples = next_alpha_bar.sqrt() * clean + (1 - next_alpha_bar).sqrt() * predicted_noise