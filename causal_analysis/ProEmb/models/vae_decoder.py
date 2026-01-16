import torch
import torch.nn as nn

class VAEDecoder(nn.Module):
    """
    Variational Decoder module. Reconstructs the proxy inputs from the latent representation (implements Eq. (11):contentReference[oaicite:16]{index=16}).
    """
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims=None):
        """
        :param latent_dim: Dimension of the latent vector (input to decoder).
        :param output_dim: Dimension of the reconstructed output (matches original input dimension).
        :param hidden_dims: List of hidden layer sizes for the decoder. If None or empty, 
                            the decoder maps directly from latent_dim to output_dim.
        """
        super(VAEDecoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = []
        # Fully connected hidden layers for decoder
        self.hidden_layers = nn.ModuleList()
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        # Output layer to reconstruct the concatenated proxies
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, z: torch.Tensor):
        """
        Forward pass: returns reconstructed proxy vector from latent z.
        """
        x = z
        for linear in self.hidden_layers:
            x = self.activation(linear(x))
        # Linear transformation to output dimension
        recon_x = self.output_layer(x)
        # (No activation here; assume regression to original proxy values)
        return recon_x
