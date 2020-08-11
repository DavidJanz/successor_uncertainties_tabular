import random

import torch
from torch import nn

_noise_level = 1e-5
_noise_enabled = True


def _argmax_stochastic(x, dim, keepdim=False):
    rand = torch.FloatTensor(*x.size()).uniform_()
    noise = (rand * 2 - 1) * _noise_level if _noise_enabled else 0
    return (x + noise).argmax(dim, keepdim=keepdim)


class Policy(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.action_size = action_size
        self.device = torch.device('cpu')

    def start_new_episode(self):
        pass

    def set_cuda(self):
        self.device = torch.device('cuda:0')
        return self

    def forward(self, state):
        raise NotImplementedError()

    def sample(self, state):
        return torch.multinomial(self(state), 1).squeeze().to(self.device)

    def _policy_size(self, state):
        if state.dim() > 1:
            batch_size, *_ = state.size()
            return tuple([batch_size, self.action_size])
        return tuple([self.action_size])


class ZeroPolicy(Policy):
    def forward(self, state):
        p = torch.zeros(self._policy_size(state))
        p.index_fill_(-1, torch.LongTensor([1] * self._policy_size(state)[0]), 1.0)
        return p


class UniformPolicy(Policy):
    def sample(self, state):
        return torch.randint(0, self.action_size, (1,),
                             dtype=torch.long, device=self.device)

    def forward(self, state):
        return torch.ones(self._policy_size(state), dtype=torch.float32) / self.action_size


class GreedyPolicy(Policy):
    def __init__(self, action_size, q_fn):
        super().__init__(action_size)
        self._q_fn = q_fn

    def sample(self, state):
        state = state.to(self.device)
        return _argmax_stochastic(self._q_fn(state), -1)

    def forward(self, state):
        state = state.unsqueeze(0) if state.dim() == 1 else state
        p = torch.zeros(self._policy_size(state))
        j = _argmax_stochastic(self._q_fn(state), -1)
        p[(range(len(j)), j)] = 1
        return p.squeeze(0) if len(p) == 1 else p


class EpsGreedyPolicy(Policy):
    def __init__(self, action_size, q_fn, eps):
        super().__init__(action_size)
        self._q_fn = q_fn
        self.eps = eps

    def sample(self, state):
        if random.random() < self.eps:
            return torch.randint(0, self.action_size, (1,),
                                 dtype=torch.long, device=self.device)
        else:
            state = state.to(self.device)
            return self._q_fn(state).argmax(-1, keepdim=True)

    def forward(self, state):
        state = state.unsqueeze(0) if state.dim() == 1 else state
        p = torch.ones(self._policy_size(state)) * self.eps / self.action_size
        j = _argmax_stochastic(self._q_fn(state), -1)
        p[(range(len(j)), j)] += 1 - self.eps
        return p.squeeze(0) if len(p) == 1 else p


class ThompsonPolicy(Policy):
    def __init__(self, action_size, q_fn, q_fn_external, weights_get_fn, uncertainty, beta, ube=None):
        super().__init__(action_size)
        self._q_fn = q_fn
        self.uncertainty = uncertainty
        self.beta = beta
        self.ube = ube

        self._q_fn_external = q_fn_external
        self._weights_get_fn = weights_get_fn
        self._weights = None

    def get_uncertainty(self, state):
        if self.ube is not None:
            return self.ube(state)
        return self.uncertainty(state)

    def start_new_episode(self):
        mu_w = self._weights_get_fn()
        sigma_w = self.uncertainty.covariance_matrix
        dist = torch.distributions.MultivariateNormal(mu_w, sigma_w)
        self._weights = dist.sample()

    def update(self, state, action):
        self.uncertainty.update(state, action)

    def sample(self, state):
        if self._weights is None:
            raise ValueError("Cannot sample action before sampling weights!")
        q_sample = self._q_fn_external(state, self._weights)
        return _argmax_stochastic(q_sample, -1)

    def forward(self, state, n=20):
        samples = torch.normal(
            self._q_fn(state).repeat(n, 1),
            self.get_uncertainty(state).sqrt().repeat(n, 1)
        ).view(n, -1, self.action_size)

        argmax_samples = torch.zeros_like(samples).scatter_(
            -1, _argmax_stochastic(samples, -1, keepdim=True), 1)

        policy_hat = argmax_samples.mean(0)
        return policy_hat


class BootstrapPolicy(GreedyPolicy):
    def __init__(self, action_size, model, next_head_fn):
        super().__init__(action_size, model)
        self._next_head_fn = next_head_fn
        self.model = model

    def start_new_episode(self):
        self._next_head_fn()
