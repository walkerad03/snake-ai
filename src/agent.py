import torch.nn as nn
from collections import namedtuple, deque
import random

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.Linear(128, 128),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.network(x)
