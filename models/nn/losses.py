from torch import nn

from .losses_functional import q_loss, \
    sf_loss, sr_loss, ube_loss

_default_discount_factor = 0.99


class QLoss(nn.Module):
    def __init__(self, model, discount_factor=_default_discount_factor):
        super().__init__()
        self._q_network = model
        self._discount_factor = discount_factor

    def forward(self, replay_tuple):
        return q_loss(replay_tuple, self._q_network, self._discount_factor)


class UBELoss(QLoss):
    def __init__(self, model, policy, discount_factor=_default_discount_factor):
        super().__init__(model, discount_factor)
        self._policy = policy

    def forward(self, replay_tuple):
        q_loss_val = super().forward(replay_tuple)
        ube_loss_val = ube_loss(replay_tuple, self._policy.ube, self._policy.uncertainty,
                                self._policy, self._discount_factor)
        return q_loss_val, ube_loss_val


class SuccessorLosses(nn.Module):
    def __init__(self, sf_network, policy, discount_factor):
        super().__init__()
        self._sf = sf_network
        self._policy = policy
        self.discount_factor = discount_factor

    def forward(self, replay_tuple):
        sfl = sf_loss(replay_tuple, self._sf.local_features, self._sf.global_features,
                      self._policy, self.discount_factor)
        srl = sr_loss(replay_tuple, self._sf.r)
        sql = q_loss(replay_tuple, self._sf.q_fn, self.discount_factor)
        return srl, sql, sfl


del _default_discount_factor
