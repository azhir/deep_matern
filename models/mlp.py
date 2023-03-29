import torch
from torch import nn
import torch.nn.functional as F


class MultilayerPerceptron(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        self.input_layer = nn.Linear(in_features, hidden_features)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for i in range(hidden_layers)])
        self.output_layer = nn.Linear(hidden_features, out_features)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        coords = self.input_layer(coords)
        coords = F.relu(coords)
        for layer in self.hidden_layers:
            coords = layer(coords)
            coords = F.relu(coords)
        output = self.output_layer(coords)
        return output


class PositionalEncoding(MultilayerPerceptron):
    def __init__(self, in_features, mapping_size, hidden_features, hidden_layers, out_features, scale=1.):
        super().__init__(2*mapping_size, hidden_features, hidden_layers, out_features)
        self.input_dim = in_features
        self.mapping_size = mapping_size
        self.B = scale * torch.randn(self.mapping_size, self.input_dim, requires_grad=False).cuda()

    def _input_mapping(self, x, B):
        if B is None:
            return x
        else:
            x_proj = (2.*torch.pi*x) @ B.T
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        coords = self._input_mapping(coords, self.B) # Map input to Fourier features
        output = super().forward(coords)
        return output



