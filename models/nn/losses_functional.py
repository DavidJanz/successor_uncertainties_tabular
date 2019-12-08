import torch
from torch.nn import functional


def _select(tensor, index, dim=-1):
    if tensor.dim() == 2:
        i = index.view(-1, 1) if index.dim() < 2 else index
    else:
        i = index.view(-1, 1, 1).expand(-1, -1, tensor.size()[-1])
    return tensor.gather(dim, i).squeeze(dim)


def _td_loss(x_t, xdiff_t, x_tp1, terminal, discount_factor):
    x_target = xdiff_t + (1 - terminal.float()) * discount_factor * x_tp1
    return functional.smooth_l1_loss(x_t, x_target)


def q_loss(replay_tuple, q_network, discount_factor):
    q_t_action = _select(q_network(replay_tuple.state_t), replay_tuple.action_t)

    with torch.no_grad():
        q_tp1_max, _ = q_network(replay_tuple.state_tp1).max(-1)

    return _td_loss(q_t_action, replay_tuple.reward_t, q_tp1_max, replay_tuple.terminal_t, discount_factor)


def sf_loss(replay_tuple, local_embedding, global_embedding, policy, discount_factor):
    sf_t = _select(global_embedding(replay_tuple.state_t), replay_tuple.action_t, 1)

    with torch.no_grad():
        rf_t = _select(local_embedding(replay_tuple.state_t), replay_tuple.action_t, 1)
        p_a_tp1 = policy(replay_tuple.state_tp1).unsqueeze(-1)

        sf_tp1 = (p_a_tp1 * global_embedding(replay_tuple.state_tp1)).sum(1)

    return _td_loss(sf_t, rf_t, sf_tp1, replay_tuple.terminal_t.unsqueeze(-1), discount_factor)


def ube_loss(replay_tuple, ube_fn, uncertainty_fn, policy, discount_factor):
    u_t = _select(ube_fn(replay_tuple.state_t), replay_tuple.action_t)

    with torch.no_grad():
        v_a = torch.stack([uncertainty_fn(s) for s in replay_tuple.state_t])
        v_t = _select(v_a, replay_tuple.action_t)
        p_a_tp1 = torch.cat([policy(s) for s in replay_tuple.state_tp1])
        ube_tp1 = ube_fn(replay_tuple.state_tp1)
        u_tp1 = (p_a_tp1 * ube_tp1).sum(1)

    return _td_loss(u_t.unsqueeze(-1), v_t.unsqueeze(-1), u_tp1.unsqueeze(-1),
                    replay_tuple.terminal_t.unsqueeze(-1), discount_factor ** 2)


def sr_loss(replay_tuple, reward_network):
    r_hat = _select(reward_network(replay_tuple.state_t), replay_tuple.action_t)
    return functional.mse_loss(r_hat, replay_tuple.reward_t)
