import torch

class ReparameterizedSampler:
    """
    Sampler that uses encoder outputs (mean and log_var) to sample a latent vector 
    using the reparameterization trick (implements Eq. (10):contentReference[oaicite:18]{index=18}).
    """
    def __call__(self, mu: torch.Tensor, log_var: torch.Tensor):
        """
        Sample z ~ N(mu, exp(log_var)) via reparameterization.
        :param mu: Mean tensor of shape (batch_size, latent_dim).
        :param log_var: Log-variance tensor of shape (batch_size, latent_dim).
        :return: Sampled latent tensor z of shape (batch_size, latent_dim).
        """
        # Compute standard deviation from log-variance
        std = torch.exp(0.5 * log_var)
        # Sample epsilon from N(0, I)
        eps = torch.randn_like(std)
        # Reparameterize: z = mu + eps * std
        return mu + eps * std
