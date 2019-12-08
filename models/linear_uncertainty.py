import torch
from torch import nn


class _OnlineVarianceABC(nn.Module):
    def __init__(self, input_size, n_action_dims, train_featuriser,
                 test_featuriser, prior_variance, bias):
        super().__init__()
        self._train_featuriser = train_featuriser
        self._test_featuriser = test_featuriser
        self._bias = bias

        _precision_matrix = 1 / prior_variance * torch.stack(
            [torch.eye(input_size + int(bias)) for _ in range(n_action_dims)])
        self.covariance_matrix = nn.Parameter(_precision_matrix, requires_grad=False)
        self.xy_prods = torch.zeros([input_size])

    def _prepend_unit(self, observation):
        if self._bias:
            return torch.cat((torch.ones(*observation.size()[:-1], 1), observation), dim=-1)
        return observation

    def _featurise_and_select(self, observation, action):
        raise NotImplementedError()

    def forward(self, observation):
        observation = observation.unsqueeze(0) if observation.dim() == 1 else observation
        observation = self._prepend_unit(self._test_featuriser(observation))
        covar = observation @ self.covariance_matrix @ observation.transpose(-1, -2)
        return self._covar2uncertainty(covar)

    def update(self, observation, action, reward=0.0):
        with torch.no_grad():
            observation = observation.detach()
            a = self.cov_mat_idx(action)

            features = self._featurise_and_select(observation, action)
            observation = self._prepend_unit(features)

            self.xy_prods += reward * observation.squeeze()
            matvec_prod = self.covariance_matrix[a] @ observation.transpose(-1, -2)
            scale = 1 + observation @ matvec_prod
            update_term = matvec_prod @ matvec_prod.transpose(-1, -2)
            scaled_update = 1 / scale * update_term
            self.covariance_matrix[a] -= scaled_update

    def mean_vector(self):
        return self.xy_prods @ self.covariance_matrix[0]


class OnlineVariance(_OnlineVarianceABC):
    def __init__(self, input_size, train_featuriser, test_featuriser, prior_variance=1.0, bias=False):
        super().__init__(input_size, 1, train_featuriser, test_featuriser, prior_variance, bias)

    def _featurise_and_select(self, observation, action):
        return self._train_featuriser(observation.unsqueeze(0))[:, action]

    @staticmethod
    def cov_mat_idx(_):
        return 0

    @staticmethod
    def _covar2uncertainty(covar):
        return covar.diagonal(dim1=-1, dim2=-2)


class OnlineVarianceMulti(_OnlineVarianceABC):
    def __init__(self, input_size, action_size, featuriser, prior_variance=1.0, bias=False):
        super().__init__(input_size, action_size, featuriser, featuriser, prior_variance, bias)

    def _featurise_and_select(self, observation, action):
        return self._train_featuriser(observation).unsqueeze(0)

    @staticmethod
    def cov_mat_idx(action):
        return action

    @staticmethod
    def _covar2uncertainty(covar):
        return covar.squeeze()
