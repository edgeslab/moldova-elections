import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    """
    Variational Encoder module. Takes concatenated ego and neighbor proxies (Z_i and Z_ngb)
    and outputs mean and log-variance of the latent distribution (implements Eq. (8)-(9):contentReference[oaicite:14]{index=14}).
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=None):
        """
        :param input_dim: Dimension of the concatenated proxy input vector.
        :param latent_dim: Dimension of the latent space (output).
        :param hidden_dims: List of hidden layer sizes for the encoder. If None or empty, 
                            the encoder maps directly to the latent dimensions.
        """
        super(VAEEncoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = []
        # Fully connected hidden layers
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        # Output layers for latent mean and log-variance
        self.mean_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Forward pass: returns (mu, log_var) for input proxy vector x.
        """
        # Pass through each hidden layer with ReLU activation
        for linear in self.hidden_layers:
            x = self.activation(linear(x))
        # Compute latent mean and log variance
        mu = self.mean_layer(x)
        log_var = self.logvar_layer(x)
        return mu, log_var
