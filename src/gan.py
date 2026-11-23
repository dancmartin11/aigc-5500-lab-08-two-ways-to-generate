'''
GAN Losses and Numerical Checks
'''

# Import the required libraries
import torch
from torch.nn import functional as F
import time

# Define the discriminator loss function
def discriminator_loss(real_logits, fake_logits):
    # Calculate the real loss
    real_loss = F.softplus(-real_logits).mean()
    fake_loss = F.softplus(fake_logits).mean()
    return real_loss + fake_loss

# Define the original generator loss function
def generator_loss_original(fake_logits):
    return -F.softplus(fake_logits).mean()

# Define the non-saturating generator loss function
def generator_loss_non_saturating(fake_logits):
    return F.softplus(-fake_logits).mean()

# Define the gradient penalty function
def gradient_penalty(critic, real_samples, fake_samples, rng = None):
    # Draw one interpolation weight per sample pair
    batch_size = real_samples.shape[0]
    weight_shape = (batch_size,) + (1,) * (real_samples.ndim - 1)
    weights = torch.rand(weight_shape, dtype = real_samples.dtype, device = real_samples.device, generator = rng)

    # Interpolate between paired samples and track gradients at those points
    interpolated = weights * real_samples + (1 - weights) * fake_samples
    interpolated = interpolated.detach().requires_grad_()

    # Differentiate the critic's scores with respect to its inputs
    scores = critic(interpolated)
    gradients, = torch.autograd.grad(outputs = scores.sum(), inputs = interpolated, create_graph = True)

    # Compute one gradient norm per sample and average the squared deviations
    gradient_norms = gradients.flatten(start_dim = 1).norm(dim = 1)
    penalty = (gradient_norms - 1).square().mean()

    # Return the mean norm for logging during critic training
    return penalty, gradient_norms.detach().mean()

# Define the function to train the digits GAN
def train_digits_gan(images, generator, discriminator, seed, use_wgan_gp):
    # Set the training budget and optimizer settings
    steps = 4000
    batch_size = 128
    noise_dim = 16
    discriminator_steps = 3 if use_wgan_gp else 1
    discriminator_lr = 5e-4 if use_wgan_gp else 2e-4

    # Keep interpolation draws separate from batch and latent noise draws
    rng = torch.Generator().manual_seed(seed)
    penalty_rng = torch.Generator().manual_seed(seed + 1000)

    # Create a separate optimizer for each network
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr = 5e-4, betas = (0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = discriminator_lr, betas = (0.5, 0.999))

    # Keep one diagnostic entry per generator step
    gradient_history = []
    score_gap_history = []
    start_time = time.perf_counter()

    for _ in range(steps):
        # Collect diagnostics across the critic updates in this step
        step_gradients = []
        step_score_gaps = []

        for _ in range(discriminator_steps):
            # Draw a real batch and generate a matching fake batch
            indices = torch.randint(len(images), (batch_size,), generator = rng)
            real_batch = images[indices]
            noise = torch.randn(batch_size, noise_dim, generator = rng)
            with torch.no_grad():
                fake_batch = generator(noise)

            # Score real and fake images with the current network
            discriminator_optimizer.zero_grad()
            real_scores = discriminator(real_batch)
            fake_scores = discriminator(fake_batch)

            if use_wgan_gp:
                # Maximize the real-minus-fake score gap with a weighted penalty
                score_gap = real_scores.mean() - fake_scores.mean()
                penalty, mean_norm = gradient_penalty(discriminator, real_batch, fake_batch, rng = penalty_rng)
                loss = -score_gap + 10 * penalty
                step_gradients.append(mean_norm.item())
                step_score_gaps.append(score_gap.detach().item())
            else:
                # Use the standard discriminator loss verified in Section 2
                loss = discriminator_loss(real_scores, fake_scores)

            # Backpropagate and update the discriminator or critic
            loss.backward()
            discriminator_optimizer.step()

        if use_wgan_gp:
            # Average the critic diagnostics for this generator step
            gradient_history.append(sum(step_gradients) / discriminator_steps)
            score_gap_history.append(sum(step_score_gaps) / discriminator_steps)

        # Freeze discriminator weights while propagating gradients to the generator
        discriminator.requires_grad_(False)
        generator_optimizer.zero_grad()
        noise = torch.randn(batch_size, noise_dim, generator = rng)
        fake_scores = discriminator(generator(noise))

        # Apply the generator objective for the selected method
        if use_wgan_gp:
            loss = -fake_scores.mean()
        else:
            loss = generator_loss_non_saturating(fake_scores)

        # Update the generator and re-enable discriminator gradients
        loss.backward()
        generator_optimizer.step()
        discriminator.requires_grad_(True)

    # Keep training time for the final comparison
    training_seconds = time.perf_counter() - start_time
    return gradient_history, score_gap_history, training_seconds