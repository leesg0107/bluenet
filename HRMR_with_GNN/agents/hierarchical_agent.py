import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.gnn import GNN
from models.hierarchical_policy import HierarchicalPolicy

class HierarchicalAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, num_options, device):
        self.device = device
        self.gnn = GNN(1, hidden_dim, hidden_dim).to(device)
        self.policy = HierarchicalPolicy(hidden_dim, hidden_dim, num_options, action_dim).to(device)
        self.optimizer = optim.Adam(list(self.gnn.parameters()) + list(self.policy.parameters()), lr=0.001)

    def act(self, state, edge_index):
        with torch.no_grad():
            batch = torch.zeros(state.size(0), dtype=torch.long, device=self.device)
            gnn_output = self.gnn(state, edge_index, batch)
            option_logits, action_mean = self.policy(gnn_output)
            action = torch.normal(action_mean, torch.ones_like(action_mean) * 0.1)
        return action.cpu().squeeze().numpy()

    def update(self, batch):
        states, edge_indices, actions, rewards, next_states, next_edge_indices, dones = batch
        
        # Convert numpy arrays to PyTorch tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        edge_indices = torch.LongTensor(np.array(edge_indices)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        batch_size = states.size(0)
        num_nodes = states.size(1)
        
        # Create batch tensor for GNN
        batch = torch.arange(batch_size, dtype=torch.long, device=self.device).repeat_interleave(num_nodes)

        # Reshape states
        states = states.view(-1, states.size(-1))

        # Process edge_indices
        edge_index = edge_indices[0]  # Assume all graphs have the same structure, so we can use the first one
        edge_index = edge_index.repeat(1, batch_size) + (torch.arange(batch_size, dtype=torch.long, device=self.device).repeat_interleave(edge_index.size(1)) * num_nodes)

        # GNN forward pass
        gnn_output = self.gnn(states, edge_index, batch)
        option_logits, action_mean = self.policy(gnn_output)

        # Compute loss (this is a simplified version, you might want to use a more sophisticated RL algorithm)
        loss = nn.MSELoss()(action_mean, actions)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
