import torch
import torch.nn.functional as F

def compute_vae_loss(reconstructed: torch.Tensor, original: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor):
    """
    Compute VAE loss = reconstruction error + KL divergence (Eq. (12):contentReference[oaicite:26]{index=26}).
    :param reconstructed: Reconstructed proxy vector (decoder output).
    :param original: Original proxy input vector.
    :param mu: Latent mean from encoder.
    :param log_var: Latent log-variance from encoder.
    :return: Scalar VAE loss.
    """
    # Reconstruction error (MSE summed over features, averaged over batch)
    recon_error = torch.sum((reconstructed - original) ** 2, dim=1).mean()
    # KL divergence between N(mu, sigma^2) and standard Normal(0,1)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
    return recon_error + kl_div

def compute_balance_loss(disc_output: torch.Tensor):
    """
    Compute the regularization loss to make discriminator output ~0.5 for all samples (Eq. (14):contentReference[oaicite:30]{index=30}).
    :param disc_output: Discriminator output probabilities for treatment=1.
    :return: Scalar balance loss.
    """
    target = torch.full_like(disc_output, 0.5)
    return ((disc_output - target) ** 2).mean()
