import math
import torch
import gpytorch
import torchvision as tv
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from .densenet_dkl import DenseNet


# Creating DenseNet model
class DenseNetFeatureExtractor(DenseNet):
    """Documentation for DenseNetFeatureExtractor

    """
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        return out


# Creating the GP Layer
# we'll be using one GP per feature, as in the SV-DKL paper. The
# outputs of these Gaussian processes will the be mixed in the softmax
# likelihood.
class GaussianProcessLayer(gpytorch.models.AbstractVariationalGP):

    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):

        variational_distribution = \
            gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=grid_size, batch_size=num_dim
            )
        variational_strategy = \
            gpytorch.variational.AdditiveGridInterpolationVariationalStrategy(
                self,
                grid_size=grid_size,
                grid_bounds=[grid_bounds],
                num_dim=num_dim,
                variational_distribution=variational_distribution,
                mixing_params=False,
                sum_output=False
            )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):

        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# Creating the DKL Model
# With both the DenseNet feature extractor and GP layer defined, we can
# put them together in a single module that simply calls one and then
# the other, much like building any Sequential neural network in PyTorch
class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(
            num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(
            features, self.grid_bounds[0], self.grid_bounds[1])
        res = self.gp_layer(features)
        return res


class DeepGP(object):
    """Documentation for DeepGP

    """
    def __init__(self, num_data, num_classes=10):
        super(DeepGP, self).__init__()
        self.num_classes = num_classes
        feature_extractor = DenseNetFeatureExtractor(
            block_config=(6, 6, 6), num_classes=self.num_classes)
        num_features = feature_extractor.classifier.in_features
        model = DKLModel(feature_extractor, num_dim=num_features)
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
            num_features=model.num_dim, num_classes=self.num_classes)
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model.gp_layer, num_data=num_data)
        return model, likelihood, mll
