import gpytorch


class ExactGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None, mean=None):
        super(ExactGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() if mean is None else mean
        self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) if kernel is None else kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        