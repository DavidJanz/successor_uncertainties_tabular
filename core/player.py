import torch

from core.replay import ReplayEpisode


def ev(s):
    return torch.argmax(s).item()


def play_train_episode(env, policy, debug=False):
    replay_episode = ReplayEpisode()
    policy.start_new_episode()

    state, terminal = env.reset(), False
    replay_episode.append(state.data)

    # print("*" * 20)

    while not terminal:
        pre_state = env.state.data
        action = policy.sample(env.state.data).cpu()

        state, reward, terminal = env.interact(action.data)
        replay_episode.append(state.data, action.data,
                              reward.data, terminal.data)

        # print(f"s: {ev(pre_state)}, a: {action} => s': {ev(state)}")

        if debug:
            a1, a2 = [a.item() for a in policy(pre_state).squeeze()]
            print(
                f"pre_state {pre_state.argmax()}; policy {a1:.2f}, {a2:.2f} -> action={action.item()}; state {state.argmax()}")

        if hasattr(policy, 'uncertainty') and policy.uncertainty is not None:
            policy.uncertainty.update(pre_state.data, action.data.item(), reward.data.item())

    return replay_episode


def play_test_episode(env, policy):
    policy.start_new_episode()
    state, terminal = env.reset(), False

    episode_reward = 0
    while not terminal:
        action = policy.sample(env.state.data).cpu()
        state, reward, terminal = env.interact(action.data)
        episode_reward += reward

    return episode_reward
