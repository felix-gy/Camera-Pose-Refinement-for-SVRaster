import torch
import torch.nn as nn


class DeferredAppearanceMLP(nn.Module):
    """
    Deferred Neural Appearance Decoder inspired by SNeRG.
    Evaluated strictly ONCE PER PIXEL on composited features, base color, and ray view direction:
    [F(r) (8D), C_base(r) (3D), d(r) (3D)] in R^14 -> MLP(14, 32, 32, 3) -> R_view(r) in (-1, 1)^3
    """
    def __init__(self, feat_dim=8, hidden_dim=32):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        in_dim = feat_dim + 3 + 3  # [F, base_rgb, viewdir]
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.0, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_uniform_(self.fc2.weight, a=0.0, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

        # Initialize near zero (std=1e-4) so initial residual is near zero,
        # starting virtually identical to SH0 but allowing immediate gradient flow.
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, feat, base_rgb, viewdir):
        """
        Args:
            feat: [..., feat_dim] accumulated latent feature
            base_rgb: [..., 3] accumulated base color
            viewdir: [..., 3] normalized camera view directions
        Returns:
            residual: [..., 3] view-dependent color residual in (-1, 1)
        """
        x = torch.cat([feat, base_rgb, viewdir], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
