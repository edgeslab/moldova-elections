import torch
import torch.nn as nn
import torch.optim as optim
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

class MetaLearner:
    """
    T-learner meta-learner for counterfactual outcome prediction:contentReference[oaicite:35]{index=35}.
    Trains separate models for treated and control units (base learners) to estimate outcomes.
    """
    def __init__(self, latent_dim: int, base_model: str = 'mlp', hidden_dims=None):
        """
        :param latent_dim: Dimension of latent representation (input features for base learners).
        :param base_model: Type of base learner ('mlp', 'linear', or 'lightgbm').
        :param hidden_dims: Hidden layer sizes for MLP (if base_model='mlp'). Ignored for linear or lightgbm.
        """
        self.base_model = base_model.lower()
        self.latent_dim = latent_dim
        # Initialize base learners for treatment=1 and treatment=0
        if self.base_model == 'mlp':
            if hidden_dims is None:
                hidden_dims = [64, 32]  # default hidden layer sizes
            # Define a simple MLP for regression
            def build_mlp():
                layers = []
                prev_dim = latent_dim
                for h in hidden_dims:
                    layers.append(nn.Linear(prev_dim, h))
                    layers.append(nn.ReLU())
                    prev_dim = h
                layers.append(nn.Linear(prev_dim, 1))  # output layer
                return nn.Sequential(*layers)
            self.model_t = build_mlp()  # model for treated group
            self.model_c = build_mlp()  # model for control group
        elif self.base_model == 'linear':
            # Simple linear regression models
            self.model_t = nn.Linear(latent_dim, 1)
            self.model_c = nn.Linear(latent_dim, 1)
        elif self.base_model == 'lightgbm':
            if lgb is None:
                raise ImportError("LightGBM is not installed.")
            # Initialize LightGBM regressors (will train outside of PyTorch autograd)
            self.model_t = lgb.LGBMRegressor()
            self.model_c = lgb.LGBMRegressor()
        else:
            raise ValueError(f"Unknown base_model type: {base_model}")

    def fit(self, X, treatment, y, num_epochs=100, lr=1e-3):
        """
        Train the treated and control outcome models on the provided data.
        :param X: Latent representations (torch.Tensor or numpy array) shape (N, latent_dim).
        :param treatment: Treatment indicator (0/1) for each sample (shape (N,)).
        :param y: Observed outcomes for each sample (shape (N,)).
        :param num_epochs: Number of epochs for training (only for neural network models).
        :param lr: Learning rate (only for neural network models).
        """
        # Convert inputs to torch tensors if not already
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if not torch.is_tensor(treatment):
            treatment = torch.tensor(treatment, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)
        # Split into treated and control subsets
        treated_idx = (treatment == 1)
        control_idx = (treatment == 0)
        X_t, y_t = X[treated_idx], y[treated_idx]
        X_c, y_c = X[control_idx], y[control_idx]
        if self.base_model in ['mlp', 'linear']:
            # Train PyTorch models using MSE loss
            optimizer_t = optim.Adam(self.model_t.parameters(), lr=lr)
            optimizer_c = optim.Adam(self.model_c.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            for epoch in range(num_epochs):
                # Update treated model
                self.model_t.train()
                optimizer_t.zero_grad()
                pred_t = self.model_t(X_t).squeeze()
                loss_t = loss_fn(pred_t, y_t)
                loss_t.backward()
                optimizer_t.step()
                # Update control model
                self.model_c.train()
                optimizer_c.zero_grad()
                pred_c = self.model_c(X_c).squeeze()
                loss_c = loss_fn(pred_c, y_c)
                loss_c.backward()
                optimizer_c.step()
        elif self.base_model == 'lightgbm':
            # Prepare numpy data for LightGBM
            X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
            t_np = treatment.detach().cpu().numpy() if torch.is_tensor(treatment) else treatment
            y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else y
            # Fit gradient boosting models on each subset
            self.model_t.fit(X_np[t_np == 1], y_np[t_np == 1])
            self.model_c.fit(X_np[t_np == 0], y_np[t_np == 0])

    def predict(self, X, treatment):
        """
        Predict outcome for given latent X and a treatment indicator.
        :param X: Latent representation(s) of shape (N, latent_dim).
        :param treatment: Treatment indicator(s) (0 or 1) for each sample.
        :return: Predicted outcome(s) as a tensor or array of shape (N,).
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if not torch.is_tensor(treatment):
            treatment = torch.tensor(treatment, dtype=torch.float32)
        # Ensure torch models are in eval mode
        if self.base_model in ['mlp', 'linear']:
            self.model_t.eval()
            self.model_c.eval()
        preds = []
        # Predict sample by sample using the appropriate model
        for i in range(X.shape[0]):
            if treatment[i].item() == 1:
                if self.base_model in ['mlp', 'linear']:
                    pred = self.model_t(X[i]).item()
                else:
                    pred = self.model_t.predict(X[i].detach().cpu().numpy().reshape(1, -1))[0]
            else:
                if self.base_model in ['mlp', 'linear']:
                    pred = self.model_c(X[i]).item()
                else:
                    pred = self.model_c.predict(X[i].detach().cpu().numpy().reshape(1, -1))[0]
            preds.append(pred)
        return torch.tensor(preds, dtype=torch.float32) if self.base_model in ['mlp', 'linear'] else preds

    def predict_potential_outcomes(self, X):
        """
        Compute potential outcomes for each sample: returns (y0, y1) for treatment=0 and treatment=1.
        :param X: Latent representation(s) of shape (N, latent_dim).
        :return: Tuple (y0, y1) of predictions for each sample if control and if treated.
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if self.base_model in ['mlp', 'linear']:
            self.model_t.eval()
            self.model_c.eval()
            with torch.no_grad():
                y1 = self.model_t(X).squeeze()  # outcome if treated
                y0 = self.model_c(X).squeeze()  # outcome if control
            return y0, y1
        else:
            X_np = X.detach().cpu().numpy()
            y1 = self.model_t.predict(X_np)
            y0 = self.model_c.predict(X_np)
            return y0, y1
