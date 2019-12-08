import numpy as np
import torch
from numpy import clip, array
from numpy import prod


class EnvironmentABC:
    def __init__(self, action_size, state_shape, max_steps):
        self.state_shape = state_shape
        self.state_size = int(prod(state_shape))
        self.action_size = action_size
        self.max_steps = max_steps

        self._state = self._reward = self._terminal = self.step = None

    def _interact(self, action):
        raise NotImplementedError()

    def interact(self, action):
        if isinstance(action, torch.Tensor):
            action = action.item()
        self.step += 1

        self._interact(action)
        return self.state, self.reward, self.terminal

    def _reset(self):
        pass

    def reset(self):
        self._state = self._reward = self.step = 0
        self._terminal = False

        self._reset()
        return self.state

    @property
    def state(self):
        return torch.tensor(self._state, dtype=torch.float32)

    @property
    def reward(self):
        return torch.tensor(self._reward, dtype=torch.float32)

    @property
    def terminal(self):
        out_of_time = (self.max_steps and self.step >= self.max_steps)
        return torch.tensor(out_of_time or self._terminal, dtype=torch.uint8)

    def default_args(self):
        return {}

    def is_solved(self, test_scores):
        if len(test_scores) < 10:
            return False
        for i in range(len(test_scores) - 10):
            if sum(test_scores[i:i + 10]) > 9.8:
                return True
        return False


class EnvironmentCountableABC(EnvironmentABC):
    def __init__(self, size, max_steps):
        self._embedding = torch.eye(size)
        super().__init__(action_size=len(self._action_lookup),
                         state_shape=(size,), max_steps=max_steps)

    @property
    def state(self):
        return self._embedding[self._state]

    def list_states(self): return [self._embedding[idx] for idx in range(0, self.state_shape[0])]

    def _interact(self, action):
        self._state = self.transition(self.state)[action]
        self._reward = not self._terminal and self._reward_lookup.get(self._state, 0)


class EnvironmentTree(EnvironmentCountableABC):
    def __init__(self, n_junctions):
        self.n_junctions = n_junctions
        self._action_lookup = [0, 1]
        state_size = n_junctions * 2 + 1
        super().__init__(state_size, max_steps=0)

        self._terminal_actions = [int(a) for a in np.random.randint(0, 2, size=self.n_junctions)]

        self._state = None
        self.current_junction = None

        self._sa_embeddings = torch.eye(state_size * 2)

    def _reset(self):
        self._state = 0
        self.current_junction = 0

    def _interact(self, action):
        if action == self._terminal_actions[self.current_junction]:
            self._terminal = True
            self._state += 1
        else:
            self._state += 2

        self._reward = 0.0
        if self._state == self.state_size - 1:
            self._reward = 1.0
            self._terminal = True

        self.current_junction += 1

    def transition(self, state):
        sa = state.argmax()
        return self._embedding[sa + 1:sa + 3, :]

    def default_args(self):
        return {"test_frequency": 10,
                "n_test_episodes": 1,
                "discount_factor": 1.0}


class ActionRemap:
    def __init__(self, state_size, action_size):
        self._map = [np.random.permutation(action_size) for _ in range(state_size)]

    def __call__(self, state_idx, action_idx):
        return self._map[state_idx][action_idx]


class EnvironmentGrid(EnvironmentCountableABC):
    def __init__(self, grid_size, verbose=False):
        self.grid_size = grid_size
        self._verbose = False

        state_size = grid_size ** 2
        max_steps = grid_size - 1

        actions = [{'name': 'left', 'move': array([-1, 1]), 'reward': 0},
                   {'name': 'right', 'move': array([1, 1]), 'reward': - 1 / (100 * max_steps)}]
        self._action_remap = ActionRemap(state_size, 2)

        self._actions = self._action_lookup = actions

        self._reward_lookup = {state_size - 1: 1.0}

        super().__init__(state_size, max_steps=max_steps)

    def transition(self, state, action):
        return self._idx1d(clip(self._idx2d(state) + self._actions[action]['move'], 0, self.grid_size - 1))

    def _interact(self, action_idx):
        action = self._action_remap(self._state, action_idx)

        prestate = self._state
        self._state = self.transition(self._state, action)
        self._reward = not self._terminal and self._actions[action]['reward'] + self._reward_lookup.get(self._state, 0)
        if self._verbose:
            print(f"s1 = {self._idx2d(prestate)}, a = {action}, "
                  f"mapped to s2 = {self._idx2d(self._state)}, r = {self._reward}")

    def _idx2d(self, idx): return array([idx // self.grid_size, idx % self.grid_size])

    def _idx1d(self, idx2d): return idx2d[0] * self.grid_size + idx2d[1]

    def default_args(self):
        return {"test_frequency": 10,
                "n_test_episodes": 1,
                "discount_factor": 1.0}
