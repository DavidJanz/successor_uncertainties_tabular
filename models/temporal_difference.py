import random

import torch
from torch import nn

import models.nn.utils
from models.nn.networks import FeatureLinear, LinearActive


class _R(nn.Module):
    def __init__(self, state_size, action_size, feature_size):
        super().__init__()
        self._get_weight_fn = None
        self._action_size, self._feature_size = action_size, feature_size

        self._net = LinearActive([state_size, feature_size, feature_size * action_size])
        models.nn.utils.folded_init(self._net)

        self._weights = nn.Parameter(torch.zeros((feature_size,)))

    def featurise(self, x):
        return self._net(x).view(-1, self._action_size, self._feature_size)

    def forward(self, x):
        return self.featurise(x) @ self.weights

    def linear(self, action_features):
        return action_features @ self.weights

    def register_weights(self, weight_fn):
        self._get_weight_fn = weight_fn

    @property
    def weights(self):
        return self._weights


class _Psi(nn.Module, models.nn.utils.CloneMixin):
    def __init__(self, state_size, action_size, feature_size, use_network):
        super().__init__()
        self.__args__ = (state_size, action_size, feature_size)
        self._action_size = action_size
        self._state_size = state_size
        self._feature_size = feature_size

        if use_network:
            self._net = LinearActive([state_size, feature_size, feature_size * action_size])
            models.nn.utils.folded_init(self._net)
        else:
            self._net = nn.Linear(state_size, feature_size * action_size, bias=False)
            nn.init.constant_(self._net.weight, 1)

    def forward(self, x):
        return self._net(x).view(-1, self._action_size, self._feature_size)


class SF(nn.Module):
    def __init__(self, state_size, action_size, feature_size, use_network):
        super().__init__()
        self.action_size, self.state_size = action_size, state_size
        self.feature_size = feature_size

        self.r = _R(state_size, action_size, self.feature_size)
        self.local_features = self.r.featurise

        self.global_features = _Psi(state_size, action_size, self.feature_size, use_network)

    def forward(self, x):
        return self.q_fn(x)

    def q_fn(self, x):
        return self.r.linear(self.global_features(x))

    def compute_q_fn_external(self, x, weights):
        q_values = self.global_features(x) @ weights.t()
        return q_values.squeeze(-1)

    def get_weights(self):
        return self.r.weights

    def register_weights(self, weight_fn):
        self.r.register_weights(weight_fn)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        self.state_size = state_size

        self.q_fn = FeatureLinear(state_size, action_size, [feature_size])
        self.local_features = self.q_fn.featurise

        nn.init.zeros_(self.q_fn.linear.weight)

        self.ube_net = nn.Linear(feature_size, action_size, bias=False)
        self.ube_net.weight.data = torch.abs(self.ube_net.weight.data)

    def ube_fn(self, x):
        with torch.no_grad():
            embedding = self.local_features(x)
        return self.ube_net(embedding)

    def next_head(self):
        pass

    def forward(self, x):
        return self.q_fn(x)

    def compute_q_fn_external(self, x, weights):
        q_values = self.local_features(x) @ weights.t()
        return q_values.squeeze(-1)

    def get_weights(self):
        return self.q_fn.linear.weight


class BootQNetwork(nn.Module):
    def __init__(self, state_size, action_size, feature_size, prior_weight=0.0, n_heads=10):
        super().__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        self.state_size = state_size

        self.prior_weight = prior_weight

        self.heads = nn.ModuleList([FeatureLinear(state_size, action_size, [feature_size]) for _ in range(n_heads)])
        self.priors = [FeatureLinear(state_size, action_size, [feature_size]) for _ in range(n_heads)]

        self.head = 0
        self._n_heads = n_heads

    def next_head(self):
        self.head = random.randint(0, self._n_heads - 1)

    def forward(self, x):
        return self.heads[self.head](x) + self.prior_weight * self.priors[self.head](x)
