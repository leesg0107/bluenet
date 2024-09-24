import numpy as np
import matplotlib.pyplot as plt

class GridWorldEnv:
    def __init__(self, grid_size=5, n_agents=2):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.agent_positions = [self._random_position() for _ in range(self.n_agents)]
        self.goal_position = self._random_position()
        self.done = False
        return self._get_obs()

    def _random_position(self):
        return np.random.randint(0, self.grid_size, size=2)

    def _get_obs(self):
        visual_range = 3
        obs = []
        for pos in self.agent_positions:
            vec = np.zeros((visual_range, visual_range, 3))
            for i in range(visual_range):
                for j in range(visual_range):
                    vec[i, j, 0] = 1.0
                    vec[i, j, 1] = 1.0
                    vec[i, j, 2] = 1.0

            # detect block
            if pos[0] - 1 >= 0 and pos[1] + 1 < self.grid_size and self.grid[pos[0] - 1, pos[1] + 1] == 1:
                vec[0, 0, 0] = 0.0
                vec[0, 0, 1] = 0.0
                vec[0, 0, 2] = 0.0
            if pos[1] + 1 < self.grid_size and self.grid[pos[0], pos[1] + 1] == 1:
                vec[0, 1, 0] = 0.0
                vec[0, 1, 1] = 0.0
                vec[0, 1, 2] = 0.0
            if pos[0] + 1 < self.grid_size and pos[1] + 1 < self.grid_size and self.grid[pos[0] + 1, pos[1] + 1] == 1:
                vec[0, 2, 0] = 0.0
                vec[0, 2, 1] = 0.0
                vec[0, 2, 2] = 0.0
            if pos[0] - 1 >= 0 and self.grid[pos[0] - 1, pos[1]] == 1:
                vec[1, 0, 0] = 0.0
                vec[1, 0, 1] = 0.0
                vec[1, 0, 2] = 0.0
            if pos[0] + 1 < self.grid_size and self.grid[pos[0] + 1, pos[1]] == 1:
                vec[1, 2, 0] = 0.0
                vec[1, 2, 1] = 0.0
                vec[1, 2, 2] = 0.0
            if pos[0] - 1 >= 0 and pos[1] - 1 >= 0 and self.grid[pos[0] - 1, pos[1] - 1] == 1:
                vec[2, 0, 0] = 0.0
                vec[2, 0, 1] = 0.0
                vec[2, 0, 2] = 0.0
            if pos[1] - 1 >= 0 and self.grid[pos[0], pos[1] - 1] == 1:
                vec[2, 1, 0] = 0.0
                vec[2, 1, 1] = 0.0
                vec[2, 1, 2] = 0.0
            if pos[0] + 1 < self.grid_size and pos[1] - 1 >= 0 and self.grid[pos[0] + 1, pos[1] - 1] == 1:
                vec[2, 2, 0] = 0.0
                vec[2, 2, 1] = 0.0
                vec[2, 2, 2] = 0.0

            # detect self
            vec[1, 1, 0] = 1.0
            vec[1, 1, 1] = 0.0
            vec[1, 1, 2] = 0.0

            # detect goal
            if np.array_equal(self.goal_position, pos + [-1, 1]):
                vec[0, 0, 0] = 0.0
                vec[0, 0, 1] = 1.0
                vec[0, 0, 2] = 0.0
            if np.array_equal(self.goal_position, pos + [0, 1]):
                vec[0, 1, 0] = 0.0
                vec[0, 1, 1] = 1.0
                vec[0, 1, 2] = 0.0
            if np.array_equal(self.goal_position, pos + [1, 1]):
                vec[0, 2, 0] = 0.0
                vec[0, 2, 1] = 1.0
                vec[0, 2, 2] = 0.0
            if np.array_equal(self.goal_position, pos + [-1, 0]):
                vec[1, 0, 0] = 0.0
                vec[1, 0, 1] = 1.0
                vec[1, 0, 2] = 0.0
            if np.array_equal(self.goal_position, pos + [1, 0]):
                vec[1, 2, 0] = 0.0
                vec[1, 2, 1] = 1.0
                vec[1, 2, 2] = 0.0
            if np.array_equal(self.goal_position, pos + [-1, -1]):
                vec[2, 0, 0] = 0.0
                vec[2, 0, 1] = 1.0
                vec[2, 0, 2] = 0.0
            if np.array_equal(self.goal_position, pos + [0, -1]):
                vec[2, 1, 0] = 0.0
                vec[2, 1, 1] = 1.0
                vec[2, 1, 2] = 0.0
            if np.array_equal(self.goal_position, pos + [1, -1]):
                vec[2, 2, 0] = 0.0
                vec[2, 2, 1] = 1.0
                vec[2, 2, 2] = 0.0

            obs.append(vec)
        return obs

    def step(self, actions):
        rewards = []
        for i, action in enumerate(actions):
            if action == 0:  # up
                self.agent_positions[i][1] = min(self.agent_positions[i][1] + 1, self.grid_size - 1)
            elif action == 1:  # down
                self.agent_positions[i][1] = max(self.agent_positions[i][1] - 1, 0)
            elif action == 2:  # left
                self.agent_positions[i][0] = max(self.agent_positions[i][0] - 1, 0)
            elif action == 3:  # right
                self.agent_positions[i][0] = min(self.agent_positions[i][0] + 1, self.grid_size - 1)

            if np.array_equal(self.agent_positions[i], self.goal_position):
                rewards.append(1)
                self.done = True
            else:
                rewards.append(0)

        return rewards, self.done

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        for pos in self.agent_positions:
            grid[tuple(pos)] = 'A'
        grid[tuple(self.goal_position)] = 'G'
        print("\n".join([" ".join(row) for row in grid]))
        print()

    def plot_scene(self):
        fig, ax = plt.subplots()
        ax.imshow(self._get_full_obs())
        plt.show()

    def _get_full_obs(self):
        obs = np.ones((self.grid_size, self.grid_size, 3))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if [i, j] == self.goal_position.tolist():
                    obs[i, j] = [0, 1, 0]  # Goal is green
                elif [i, j] in [pos.tolist() for pos in self.agent_positions]:
                    obs[i, j] = [1, 0, 0]  # Agents are red
                else:
                    obs[i, j] = [1, 1, 1]  # Empty space is white
        return obs
