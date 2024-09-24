from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v3

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Actor(nn.Module):
    def __init__(self, obs_size, N_action):
        super(Actor, self).__init__()
        self.N_action = N_action
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.N_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

    def get_action(self, obs):
        probs = self.forward(obs)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), probs

class Critic(nn.Module):
    def __init__(self, obs_size, N_action):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, N_action * N_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_value(self, s1, s2):
        x = torch.cat([s1, s2], dim=-1)
        return self.forward(x)

class COMA(object):
    def __init__(self, obs_size_1, obs_size_2, N_action):
        self.N_action = N_action
        self.actor1 = Actor(obs_size_1, self.N_action)
        self.actor2 = Actor(obs_size_2, self.N_action)
        self.critic = Critic(max(obs_size_1, obs_size_2), self.N_action)
        self.gamma = 0.95
        self.c_loss_fn = torch.nn.MSELoss()

    def get_action(self, obs1, obs2):
        obs1 = torch.tensor(obs1, dtype=torch.float32).unsqueeze(0)
        obs2 = torch.tensor(obs2, dtype=torch.float32).unsqueeze(0)
        action1, pi_a1 = self.actor1.get_action(obs1)
        action2, pi_a2 = self.actor2.get_action(obs2)
        return action1, pi_a1, action2, pi_a2

    def cross_prod(self, pi_a1, pi_a2):
        new_pi = torch.zeros(1, self.N_action * self.N_action)
        for i in range(self.N_action):
            for j in range(self.N_action):
                new_pi[0, i * self.N_action + j] = pi_a1[0, i] * pi_a2[0, j]
        return new_pi

    def train(self, o1_list, a1_list, pi_a1_list, o2_list, a2_list, pi_a2_list, r_list):
        a1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=3e-4)
        a2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=3e-4)
        c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        T = min(len(r_list), len(a1_list), len(a2_list), len(pi_a1_list), len(pi_a2_list))
        print(f"T: {T}")
        print(f"r_list length: {len(r_list)}")
        print(f"a1_list length: {len(a1_list)}")
        print(f"a2_list length: {len(a2_list)}")
        print(f"pi_a1_list length: {len(pi_a1_list)}")
        print(f"pi_a2_list length: {len(pi_a2_list)}")

        obs1 = torch.stack([torch.tensor(o, dtype=torch.float32) for o in o1_list[:T]])
        obs2 = torch.stack([torch.tensor(o, dtype=torch.float32) for o in o2_list[:T]])

        Q = self.critic.get_value(obs1, obs2)
        Q_est = Q.clone()
        
        print(f"Q shape: {Q.shape}")
        print(f"Q_est shape: {Q_est.shape}")
        
        for t in range(T - 1):
            a_index = a1_list[t] * self.N_action + a2_list[t]
            Q_est[t, a_index] = r_list[t] + self.gamma * torch.sum(self.cross_prod(pi_a1_list[t], pi_a2_list[t]) * Q[t, :])
        
        a_index = a1_list[T-1] * self.N_action + a2_list[T-1]
        Q_est[T-1, a_index] = r_list[T-1]

        c_loss = self.c_loss_fn(Q, Q_est.detach())
        c_optimizer.zero_grad()
        c_loss.backward()
        c_optimizer.step()

        A1_list = []
        A2_list = []
        for t in range(T):
            temp_Q1 = torch.zeros(1, self.N_action)
            temp_Q2 = torch.zeros(1, self.N_action)
            for a1 in range(self.N_action):
                temp_Q1[0, a1] = Q[t, a1 * self.N_action + a2_list[t]]
            for a2 in range(self.N_action):
                temp_Q2[0, a2] = Q[t, a1_list[t] * self.N_action + a2]
            
            a_index = a1_list[t] * self.N_action + a2_list[t]
            temp_A1 = Q[t, a_index] - torch.sum(pi_a1_list[t] * temp_Q1)
            temp_A2 = Q[t, a_index] - torch.sum(pi_a2_list[t] * temp_Q2)
            A1_list.append(temp_A1)
            A2_list.append(temp_A2)

        a1_loss = torch.FloatTensor([0.0])
        a2_loss = torch.FloatTensor([0.0])
        for t in range(T):
            a1_loss = a1_loss + A1_list[t].item() * torch.log(pi_a1_list[t][0, a1_list[t]])
            a2_loss = a2_loss + A2_list[t].item() * torch.log(pi_a2_list[t][0, a2_list[t]])
        
        a1_loss = -a1_loss / T
        a2_loss = -a2_loss / T

        a1_optimizer.zero_grad()
        a1_loss.backward()
        a1_optimizer.step()

        a2_optimizer.zero_grad()
        a2_loss.backward()
        a2_optimizer.step()


if __name__ == '__main__':
    torch.set_num_threads(1)
    env = simple_tag_v3.env()
    env.reset()
    max_epi_iter = 1000
    max_MC_iter = 200
    obs_size_1 = env.observation_space('adversary_0').shape[0]
    obs_size_2 = env.observation_space('adversary_1').shape[0]
    action_size = env.action_space('adversary_0').n

    agent = COMA(obs_size_1, obs_size_2, N_action=5)  # Assuming 5 actions in simple_tag
    train_curve = []
    for epi_iter in range(max_epi_iter):
        env.reset()
        o1_list = []
        a1_list = []
        pi_a1_list = []
        o2_list = []
        a2_list = []
        pi_a2_list = []
        r_list = []
        acc_r = 0
        MC_iter = 0
        for _ in range(max_MC_iter):
            for agent_name in env.agent_iter():
                obs, reward, termination, truncation, _ = env.last()
                done = termination or truncation
                if done:
                    action = None
                else:
                    if agent_name == 'adversary_0':
                        o1_list.append(obs)
                        action1, pi_a1 = agent.actor1.get_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                        a1_list.append(action1)
                        pi_a1_list.append(pi_a1)
                        action = action1
                    elif agent_name == 'adversary_1':
                        o2_list.append(obs)
                        action2, pi_a2 = agent.actor2.get_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                        a2_list.append(action2)
                        pi_a2_list.append(pi_a2)
                        action = action2
                    else:
                        action = env.action_space(agent_name).sample()  # Random action for other agents
                
                env.step(action)
                acc_r += reward
                r_list.append(reward)
                
                MC_iter += 1
                if done:
                    break
            if done:
                break
        
        if MC_iter > 0 and epi_iter % 10 == 0:
            train_curve.append(acc_r / MC_iter)
        print('Episode', epi_iter, 'reward', acc_r / MC_iter if MC_iter > 0 else 0)
        if MC_iter > 0:
            agent.train(o1_list, a1_list, pi_a1_list, o2_list, a2_list, pi_a2_list, r_list)
    
    plt.plot(train_curve, linewidth=1, label='COMA')
    plt.show()
