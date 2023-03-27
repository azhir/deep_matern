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
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        coords = self.input_layer(coords)
        coords = F.relu(coords)
        for layer in self.hidden_layers:
            coords = layer(coords)
            coords = F.relu(coords)
        output = self.output_layer(coords)
        return output

        