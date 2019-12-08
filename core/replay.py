import random
from collections import deque
from collections import namedtuple

import numpy as np
import torch

ReplayTuple = namedtuple('ReplayTuple',
                         ['state_t', 'action_t', 'state_tp1', 'reward_t', 'terminal_t'])


class UniformDataset:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def __len__(self):
        return len(self.buffer)

    def append(self, episode):
        for item in episode:
            print(f"ADDING {item}")
            self.buffer.append(item)

    def __getitem__(self, item):
        return self.buffer[item]


class UniformDatasetOneHot(UniformDataset):
    def __init__(self, state_size, max_size=1000):
        super().__init__(max_size)
        self.embeddings = torch.eye(state_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, episode):
        for (s, a, sn, r, d) in episode:
            self.buffer.append((torch.argmax(s), a, torch.argmax(sn), r, d))

    def __getitem__(self, item):
        (s, a, sn, r, d) = self.buffer[item]
        return self.embeddings[s], a, self.embeddings[sn], r, d


class EnsembleDataset:
    def __init__(self, n_ensembles, max_size=1000, p=0.5):
        self.buffers = [deque(maxlen=max_size) for _ in range(n_ensembles)]
        self.p = p

    def len(self, head):
        return len(self.buffers[head])

    def append(self, episode):
        for item in episode:
            for buffer in self.buffers:
                if random.random() < self.p:
                    buffer.append(item)

    def __getitem__(self, item):
        head, element = item
        return self.buffers[head][element]


class ReplayEpisode:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminals = []

    def __len__(self):
        return len(self.terminals)

    def __getitem__(self, item):
        return (self.states[item], self.actions[item], self.states[item + 1],
                self.rewards[item], self.terminals[item])

    def append(self, state, action=None, reward=None, terminal=None):
        self.states.append(state.data)

        if action is not None:
            self.actions.append(action.data)
        if reward is not None:
            self.rewards.append(reward.data)
        if terminal is not None:
            self.terminals.append(terminal.data)

    def as_tuple(self):
        return ReplayTuple(
            *map(torch.stack, [self.states[:-1], self.actions,
                               self.states[1:], self.rewards, self.terminals]))

    @property
    def reward(self):
        return sum(self.rewards).item()


def as_replaytuple(batch, device='cpu'):
    return ReplayTuple(*[torch.stack(x).to(device) for x in zip(*batch)])


def make_sample_function(dataset, batch_size, algorithm):
    if algorithm == 'boot':
        def sample_replay_batch(head):
            if dataset.len(head):
                return as_replaytuple((dataset[(head, i)] for i in
                                       np.random.randint(0, dataset.len(head), batch_size)))
            else:
                return None
    else:
        def sample_replay_batch():
            return as_replaytuple((dataset[i] for i in
                                   np.random.randint(0, len(dataset), batch_size)))

    return sample_replay_batch
