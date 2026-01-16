import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from models.vae_encoder import VAEEncoder
from models.vae_decoder import VAEDecoder
from models.discriminator import Discriminator
from models.sampler import ReparameterizedSampler
from loss.losses import compute_vae_loss, compute_balance_loss
from models.meta_learner import MetaLearner

class Trainer:
    """
    Coordinates training of encoder, decoder, discriminator, and meta-learner.
    Handles alternating adversarial training and outcome model fitting.
    """
    def __init__(self, input_dim: int, latent_dim: int, 
                 encoder_hidden=None, decoder_hidden=None, disc_hidden=None,
                 base_model='mlp', base_hidden=None,
                 batch_size=64, lr_vae=1e-3, lr_disc=1e-3, device=None):
        """
        :param input_dim: Dimensionality of concatenated proxies (encoder input).
        :param latent_dim: Dimensionality of the latent embedding space.
        :param encoder_hidden: Hidden layer sizes for VAEEncoder.
        :param decoder_hidden: Hidden layer sizes for VAEDecoder.
        :param disc_hidden: Hidden layer sizes for Discriminator.
        :param base_model: Base learner type for MetaLearner ('mlp', 'linear', 'lightgbm').
        :param base_hidden: Hidden layer sizes for meta-learner MLP (if applicable).
        :param batch_size: Training batch size.
        :param lr_vae: Learning rate for VAE (encoder+decoder).
        :param lr_disc: Learning rate for discriminator.
        :param device: Device ('cpu' or 'cuda'). Defaults to CUDA if available.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize components
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dims=encoder_hidden).to(self.device)
        self.decoder = VAEDecoder(latent_dim, input_dim, hidden_dims=decoder_hidden).to(self.device)
        self.discriminator = Discriminator(latent_dim, hidden_dims=disc_hidden).to(self.device)
        self.sampler = ReparameterizedSampler()
        self.meta_learner = MetaLearner(latent_dim, base_model=base_model, hidden_dims=base_hidden)
        # Optimizers for VAE (encoder+decoder jointly) and discriminator
        self.optim_vae = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr_vae)
        self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc)
        self.batch_size = batch_size

    def load_data(self, csv_path, ego_proxy_cols, ngb_proxy_cols, treat_col, outcome_col):
        """
        Load dataset from CSV file and return as TensorDataset.
        Expects columns for ego proxies, neighbor proxies, treatment indicator, and outcome.
        """
        df = pd.read_csv(csv_path)
        # Concatenate ego and neighbor proxy features
        Z_i = torch.tensor(df[ego_proxy_cols].values, dtype=torch.float32)
        Z_ngb = torch.tensor(df[ngb_proxy_cols].values, dtype=torch.float32)
        X = torch.cat([Z_i, Z_ngb], dim=1)
        treatment = torch.tensor(df[treat_col].values, dtype=torch.float32)
        outcome = torch.tensor(df[outcome_col].values, dtype=torch.float32)
        return TensorDataset(X, treatment, outcome)

    def train_embedding(self, dataset, epochs=50):
        """
        Train the VAE (encoder+decoder) and discriminator adversarially to learn balanced embeddings.
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(epochs):
            for X_batch, treatment_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                treatment_batch = treatment_batch.to(self.device)
                # ---- Update Discriminator ----
                self.encoder.eval()
                self.decoder.eval()
                self.discriminator.train()
                # Encode and sample latent (detach to fix encoder during discriminator update)
                mu, log_var = self.encoder(X_batch)
                z = self.sampler(mu, log_var).detach()
                # Discriminator forward pass and loss (binary cross-entropy)
                pred = self.discriminator(z).view(-1)
                disc_loss = torch.nn.functional.binary_cross_entropy(pred, treatment_batch)
                # Optimize discriminator
                self.optim_disc.zero_grad()
                disc_loss.backward()
                self.optim_disc.step()
                # ---- Update VAE (Encoder+Decoder) ----
                self.encoder.train()
                self.decoder.train()
                # Encode and sample latent (for encoder update, keep grad on encoder)
                mu, log_var = self.encoder(X_batch)
                z = self.sampler(mu, log_var)  # gradients flow into encoder through z
                recon_X = self.decoder(z)
                # Compute VAE reconstruction+KL loss
                vae_loss = compute_vae_loss(recon_X, X_batch, mu, log_var)
                # Compute adversarial regularization loss (freeze discriminator params during this)
                for param in self.discriminator.parameters():
                    param.requires_grad_(False)
                d_out = self.discriminator(z)  # get discriminator prediction without updating D
                balance_loss = compute_balance_loss(d_out)
                for param in self.discriminator.parameters():
                    param.requires_grad_(True)
                # Total encoder loss and optimize (encoder+decoder)
                total_loss = vae_loss + balance_loss
                self.optim_vae.zero_grad()
                total_loss.backward()
                self.optim_vae.step()
            # (Optional) print loss values for monitoring training progress
        # Set models to eval mode after training
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

    def train_outcome_model(self, dataset, epochs=100, lr=1e-3):
        """
        Train the meta-learner outcome models on the latent embeddings of the dataset.
        """
        X, treatment, outcome = dataset.tensors
        X = X.to(self.device)
        # Encode all data to latent space (using mean as deterministic embedding)
        with torch.no_grad():
            mu, log_var = self.encoder(X)
            Z = mu  # use mean of distribution as representation for outcome model
        # Train the outcome models (meta-learner) on observed outcomes
        self.meta_learner.fit(Z.cpu(), treatment.cpu(), outcome.cpu(), num_epochs=epochs, lr=lr)

    def train_all(self, dataset, embed_epochs=50, outcome_epochs=100):
        """
        Full training pipeline: first learn embeddings (with adversarial balance), then train outcome model.
        """
        # Phase 1: VAE + adversarial training
        self.train_embedding(dataset, epochs=embed_epochs)
        # Phase 2: Outcome model training on balanced latent representation
        self.train_outcome_model(dataset, epochs=outcome_epochs)

    def get_trained_components(self):
        """Return the trained encoder, decoder, discriminator, and meta_learner."""
        return self.encoder, self.decoder, self.discriminator, self.meta_learner
