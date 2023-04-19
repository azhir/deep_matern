import torch
from torch import nn
import torch.nn.functional as F

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device 

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
        self.B = scale * torch.randn(self.mapping_size, self.input_dim, requires_grad=False).to(get_device())

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



class Deep_Matern(MultilayerPerceptron):

    def __init__(self, in_features, mapping_size, hidden_features, hidden_layers, out_features, scale=1.):
        super().__init__(2*mapping_size, hidden_features, hidden_layers, out_features)
        self.input_dim = in_features
        self.mapping_size = mapping_size
        self.B = scale * torch.randn(self.mapping_size, self.input_dim, requires_grad=False).to(get_device())
        
    def matern_activation(x, nu_ind=2, ell=0.5):
        """
        x: Input to the activation function
        device: A torch.device object
        nu_ind: Index for choosing Matern smoothness (look at nu_list below)
        ell: Matern length-scale, only 0.5 and 1 available with precalculated scaling coefficients
        """
        nu_list = [1/2, 3/2, 5/2, 7/2, 9/2] #list of available smoothness parameters
        nu = torch.tensor(nu_list[nu_ind]) #smoothness parameter
        lamb =  torch.sqrt(2*nu)/ell #lambda parameter
        v = nu+1/2
        # Precalculated scaling coefficients for two different lengthscales (q divided by Gammafunction at nu + 0.5)
        ell05A = [4.0, 19.595917942265423, 65.31972647421809, 176.69358285524189, 413.0710073859664]
        ell1A = [2.0, 4.898979485566356, 8.16496580927726, 11.043348928452618, 12.90846898081145]
        if ell == 0.5:
            A = ell05A[nu_ind]
        if ell == 1:
            A = ell1A[nu_ind]
        y = A*torch.sign(x)*torch.abs(x)**(v-1)*torch.exp(-lamb*torch.abs(x))
        y[x<0] = 0 # Values at x<0 must all be 0
        return y


    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        coords = self.input_layer(coords)
        coords = self.matern_activation(coords)
        for layer in self.hidden_layers:
            coords = layer(coords)
            coords = self.matern_activation(coords)
        output = self.output_layer(coords)
        return output