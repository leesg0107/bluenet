import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.gnn import GNN
from models.agent_policy import AgentPolicy

class DistributedAgentSystem:
    def __init__(self, input_dim, hidden_dim, num_options, action_dim, device):
        self.device = device
        self.gnn = GNN(input_dim, hidden_dim, hidden_dim).to(device)
        self.agent_policy = AgentPolicy(hidden_dim, hidden_dim, num_options, action_dim).to(device)
        self.optimizer = optim.Adam(list(self.gnn.parameters()) + list(self.agent_policy.parameters()), lr=0.001)

    def act(self, states, edge_index):
        with torch.no_grad():
            gnn_output = self.gnn(states, edge_index, None)  # batch is None as we process all nodes at once
            option_logits, action_means = self.agent_policy(gnn_output)
            actions = torch.normal(action_means, torch.ones_like(action_means) * 0.1)
        return actions.cpu().numpy()

    def update(self, states, edge_index, actions, rewards):
        states = torch.FloatTensor(states).to(self.device)
        edge_index = torch.LongTensor(edge_index).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        gnn_output = self.gnn(states, edge_index, None)
        option_logits, action_means = self.agent_policy(gnn_output)

        # Simple policy gradient loss (you might want to use a more sophisticated RL algorithm)
        log_probs = -0.5 * ((actions - action_means) ** 2).sum(dim=-1)
        loss = -(log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def add_agent(self, new_state):
        # Logic to add a new agent (you'll need to update edge_index as well)
        pass

    def remove_agent(self, agent_index):
        # Logic to remove an agent (you'll need to update edge_index as well)
        pass
