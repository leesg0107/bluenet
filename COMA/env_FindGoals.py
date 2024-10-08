import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class EnvFindGoals(object):

    def __init__(self):
        self.start1 = [3, 1]
        self.start2 = [6, 1]
        self.dest1 = [8, 2]
        self.dest2 = [1, 2]
        self.agt1_pos = [3, 1]
        self.agt2_pos = [6, 1]
        self.occupancy = [[1, 1, 1, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 1, 1]]

    def list_add(self, a, b):
        c = [a[i] + b[i] for i in range(min(len(a), len(b)))]
        return c

    def get_agt1_obs(self):
        visual_range = 5
        vec = np.zeros((visual_range, visual_range, 3))

        for i in range(visual_range):
            for j in range(visual_range):
                vec[i, j, 0] = 1.0
                vec[i, j, 1] = 1.0
                vec[i, j, 2] = 1.0

        # detect block
        if self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1] + 1] == 1:
            vec[0, 0, 0] = 0.0
            vec[0, 0, 1] = 0.0
            vec[0, 0, 2] = 0.0
        if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] + 1] == 1:
            vec[0, 1, 0] = 0.0
            vec[0, 1, 1] = 0.0
            vec[0, 1, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1] + 1] == 1:
            vec[0, 2, 0] = 0.0
            vec[0, 2, 1] = 0.0
            vec[0, 2, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1]] == 1:
            vec[1, 0, 0] = 0.0
            vec[1, 0, 1] = 0.0
            vec[1, 0, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1]] == 1:
            vec[1, 2, 0] = 0.0
            vec[1, 2, 1] = 0.0
            vec[1, 2, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1] - 1] == 1:
            vec[2, 0, 0] = 0.0
            vec[2, 0, 1] = 0.0
            vec[2, 0, 2] = 0.0
        if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] - 1] == 1:
            vec[2, 1, 0] = 0.0
            vec[2, 1, 1] = 0.0
            vec[2, 1, 2] = 0.0
        if self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1] - 1] == 1:
            vec[2, 2, 0] = 0.0
            vec[2, 2, 1] = 0.0
            vec[2, 2, 2] = 0.0

        # detect self
        vec[1, 1, 0] = 1.0
        vec[1, 1, 1] = 0.0
        vec[1, 1, 2] = 0.0

        # detect agent2
        if self.agt2_pos == self.list_add(self.agt1_pos, [-1, 1]):
            vec[0, 0, 0] = 0.0
            vec[0, 0, 1] = 0.0
            vec[0, 0, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [0, 1]):
            vec[0, 1, 0] = 0.0
            vec[0, 1, 1] = 0.0
            vec[0, 1, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [1, 1]):
            vec[0, 2, 0] = 0.0
            vec[0, 2, 1] = 0.0
            vec[0, 2, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [-1, 0]):
            vec[1, 0, 0] = 0.0
            vec[1, 0, 1] = 0.0
            vec[1, 0, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [1, 0]):
            vec[1, 2, 0] = 0.0
            vec[1, 2, 1] = 0.0
            vec[1, 2, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [-1, -1]):
            vec[2, 0, 0] = 0.0
            vec[2, 0, 1] = 0.0
            vec[2, 0, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [0, -1]):
            vec[2, 1, 0] = 0.0
            vec[2, 1, 1] = 0.0
            vec[2, 1, 2] = 1.0
        if self.agt2_pos == self.list_add(self.agt1_pos, [1, -1]):
            vec[2, 2, 0] = 0.0
            vec[2, 2, 1] = 0.0
            vec[2, 2, 2] = 1.0
        return vec

    def get_agt2_obs(self):
        visual_range = 3
        vec = np.zeros((visual_range, visual_range, 3))

        for i in range(visual_range):
            for j in range(visual_range):
                vec[i, j, 0] = 1.0
                vec[i, j, 1] = 1.0
                vec[i, j, 2] = 1.0

        # detect block
        if self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1] + 1] == 1:
            vec[0, 0, 0] = 0.0
            vec[0, 0, 1] = 0.0
            vec[0, 0, 2] = 0.0
        if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] + 1] == 1:
            vec[0, 1, 0] = 0.0
            vec[0, 1, 1] = 0.0
            vec[0, 1, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1] + 1] == 1:
            vec[0, 2, 0] = 0.0
            vec[0, 2, 1] = 0.0
            vec[0, 2, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1]] == 1:
            vec[1, 0, 0] = 0.0
            vec[1, 0, 1] = 0.0
            vec[1, 0, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1]] == 1:
            vec[1, 2, 0] = 0.0
            vec[1, 2, 1] = 0.0
            vec[1, 2, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1] - 1] == 1:
            vec[2, 0, 0] = 0.0
            vec[2, 0, 1] = 0.0
            vec[2, 0, 2] = 0.0
        if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] - 1] == 1:
            vec[2, 1, 0] = 0.0
            vec[2, 1, 1] = 0.0
            vec[2, 1, 2] = 0.0
        if self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1] - 1] == 1:
            vec[2, 2, 0] = 0.0
            vec[2, 2, 1] = 0.0
            vec[2, 2, 2] = 0.0

        # detect self
        vec[1, 1, 0] = 0.0
        vec[1, 1, 1] = 0.0
        vec[1, 1, 2] = 1.0

        # detect agent2
        if self.agt1_pos == self.list_add(self.agt2_pos, [-1, 1]):
            vec[0, 0, 0] = 1.0
            vec[0, 0, 1] = 0.0
            vec[0, 0, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [0, 1]):
            vec[0, 1, 0] = 1.0
            vec[0, 1, 1] = 0.0
            vec[0, 1, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [1, 1]):
            vec[0, 2, 0] = 1.0
            vec[0, 2, 1] = 0.0
            vec[0, 2, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [-1, 0]):
            vec[1, 0, 0] = 1.0
            vec[1, 0, 1] = 0.0
            vec[1, 0, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [1, 0]):
            vec[1, 2, 0] = 1.0
            vec[1, 2, 1] = 0.0
            vec[1, 2, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [-1, -1]):
            vec[2, 0, 0] = 1.0
            vec[2, 0, 1] = 0.0
            vec[2, 0, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [0, -1]):
            vec[2, 1, 0] = 1.0
            vec[2, 1, 1] = 0.0
            vec[2, 1, 2] = 0.0
        if self.agt1_pos == self.list_add(self.agt2_pos, [1, -1]):
            vec[2, 2, 0] = 1.0
            vec[2, 2, 1] = 0.0
            vec[2, 2, 2] = 0.0
        return vec

    def get_full_obs(self):
        obs = np.ones((4, 10, 3))
        for i in range(4):
            for j in range(10):
                if self.occupancy[j][i] == 1:
                    obs[3-i, j, 0] = 0
                    obs[3-i, j, 1] = 0
                    obs[3-i, j, 2] = 0
                if [j, i] == self.agt1_pos:
                    obs[3-i, j, 0] = 1
                    obs[3-i, j, 1] = 0
                    obs[3-i, j, 2] = 0
                if [j, i] == self.agt2_pos:
                    obs[3-i, j, 0] = 0
                    obs[3-i, j, 1] = 0
                    obs[3-i, j, 2] = 1
        return obs

    def get_obs(self):
        return [self.get_agt1_obs(), self.get_agt2_obs()]

    def step(self, action_list):
        reward_1 = 0
        reward_2 = 0
        self.start1 = [3, 1]
        self.start2 = [6, 1]
        self.dest1 = [8, 2]
        self.dest2 = [1, 2]
        # agent1 move
        if action_list[0] == 0:    # move up
            if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] + 1] != 1:     # if can move
                self.agt1_pos[1] = self.agt1_pos[1] + 1
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] - 1] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
        elif action_list[0] == 1:  # move down
            if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] - 1] != 1:  # if can move
                self.agt1_pos[1] = self.agt1_pos[1] - 1
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] + 1] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
        elif action_list[0] == 2:  # move left
            if self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1]] != 1:  # if can move
                self.agt1_pos[0] = self.agt1_pos[0] - 1
                self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1]] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
        elif action_list[0] == 3:  # move right
            if self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1]] != 1:  # if can move
                self.agt1_pos[0] = self.agt1_pos[0] + 1
                self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1]] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1

        # agent2 move
        if action_list[1] == 0:    # move up
            if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] + 1] != 1:     # if can move
                self.agt2_pos[1] = self.agt2_pos[1] + 1
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] - 1] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
        elif action_list[1] == 1:  # move down
            if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] - 1] != 1:  # if can move
                self.agt2_pos[1] = self.agt2_pos[1] - 1
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1] + 1] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
        elif action_list[1] == 2:  # move left
            if self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1]] != 1:  # if can move
                self.agt2_pos[0] = self.agt2_pos[0] - 1
                self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1]] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
        elif action_list[1] == 3:  # move right
            if self.occupancy[self.agt2_pos[0] + 1][self.agt2_pos[1]] != 1:  # if can move
                self.agt2_pos[0] = self.agt2_pos[0] + 1
                self.occupancy[self.agt2_pos[0] - 1][self.agt2_pos[1]] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1

        if self.agt1_pos == self.dest1:
            self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 0
            self.agt1_pos = self.start1
            self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            reward_1 = reward_1 + 10

        if self.agt2_pos == self.dest2:
            self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 0
            self.agt2_pos = self.start2
            self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            reward_2 = reward_2 + 10

        done = False
        if reward_1 > 0:
            done = True
            self.reset()
        return [reward_1, reward_2], done

    def reset(self):
        self.agt1_pos = [3, 1]
        self.agt2_pos = [6, 1]

        self.occupancy = [[1, 1, 1, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 0, 1],
                          [1, 1, 1, 1]]

    def plot_scene(self):
        fig, ax = plt.subplots()
        ax.imshow(self.get_full_obs())
        plt.show()

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xticks(np.arange(-.5, 10, 1))
        ax.set_yticks(np.arange(-.5, 4, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='black', linestyle='-', linewidth=2)

        for i in range(10):
            for j in range(4):
                if self.occupancy[i][j] == 1:
                    ax.add_patch(plt.Rectangle((i, 3-j), 1, 1, color='black'))
        ax.add_patch(plt.Rectangle((self.agt2_pos[0], 3-self.agt2_pos[1]), 1, 1, color='blue'))
        ax.add_patch(plt.Rectangle((self.agt1_pos[0], 3-self.agt1_pos[1]), 1, 1, color='red'))
        plt.show()