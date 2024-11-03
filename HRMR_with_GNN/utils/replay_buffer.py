import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, edge_index, action, reward, next_state, next_edge_index, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, edge_index, action, reward, next_state, next_edge_index, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, edge_index, action, reward, next_state, next_edge_index, done = map(np.stack, zip(*batch))
        return state, edge_index, action, reward, next_state, next_edge_index, done

    def __len__(self):
        return len(self.buffer)
