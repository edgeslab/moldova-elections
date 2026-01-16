import torch
import pandas as pd
from models.vae_encoder import VAEEncoder
from models.sampler import ReparameterizedSampler
from models.meta_learner import MetaLearner

class Inferencer:
    """
    Uses trained encoder and meta-learner to infer potential outcomes, ITE (contagion effect), and ACE.
    """
    def __init__(self, encoder: VAEEncoder, meta_learner: MetaLearner):
        """
        :param encoder: Trained VAEEncoder.
        :param meta_learner: Trained MetaLearner.
        """
        self.encoder = encoder
        self.meta_learner = meta_learner
        self.sampler = ReparameterizedSampler()
        # By default, use mean of latent distribution for embedding (for less randomness in inference)
        self.use_mean = True

    def infer_from_data(self, X):
        """
        Compute potential outcomes and effects for given proxy input data.
        :param X: Proxy input features (tensor or ndarray) of shape (N, input_dim).
        :return: Dictionary with keys 'y0', 'y1', 'ite', and 'ace'.
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(next(self.encoder.parameters()).device)  # move to same device as encoder
        # Encode inputs to latent space
        self.encoder.eval()
        with torch.no_grad():
            mu, log_var = self.encoder(X)
        z = mu if self.use_mean else self.sampler(mu, log_var)  # choose mean or sample
        # Predict potential outcomes using the meta-learner
        y0, y1 = self.meta_learner.predict_potential_outcomes(z.cpu())
        # Convert outputs to torch tensors if they are numpy (LightGBM case)
        if not torch.is_tensor(y0):
            y0 = torch.tensor(y0, dtype=torch.float32)
            y1 = torch.tensor(y1, dtype=torch.float32)
        # Individual effects and average effect
        ite = y1 - y0
        ace = torch.mean(ite)
        return {'y0': y0, 'y1': y1, 'ite': ite, 'ace': ace}

    def infer_from_csv(self, csv_path, ego_proxy_cols, ngb_proxy_cols):
        """
        Load proxy features from a CSV and compute outcomes/effects.
        Assumes the CSV contains only proxy features (and possibly treatment, which is not needed for inference).
        """
        df = pd.read_csv(csv_path)
        Z_i = df[ego_proxy_cols].values
        Z_ngb = df[ngb_proxy_cols].values
        import numpy as np
        X_arr = np.concatenate([Z_i, Z_ngb], axis=1)
        X = torch.tensor(X_arr, dtype=torch.float32)
        return self.infer_from_data(X)
