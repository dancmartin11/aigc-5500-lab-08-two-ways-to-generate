'''
GAN Losses and Numerical Checks
'''

# Import the required libraries
from torch.nn import functional as F

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