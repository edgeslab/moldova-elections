import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Adversarial Discriminator that predicts the treatment from the latent vector (implements Eq. (13):contentReference[oaicite:22]{index=22}).
    """
    def __init__(self, input_dim: int, hidden_dims=None):
        """
        :param input_dim: Dimension of the latent vector (encoder output).
        :param hidden_dims: List of hidden layer sizes. If None or empty, use a single linear layer.
        """
        super(Discriminator, self).__init__()
        if hidden_dims is None:
            hidden_dims = []
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        # Final layer for binary classification (single output neuron)
        self.output_layer = nn.Linear(prev_dim, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor):
        """
        Forward pass: returns probability that treatment=1 for latent input z.
        """
        x = z
        for linear in self.hidden_layers:
            x = self.activation(linear(x))
        logit = self.output_layer(x)
        prob = self.sigmoid(logit)  # convert to [0,1] probability
        return prob
