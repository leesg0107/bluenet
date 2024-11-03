import torch
import torch.nn as nn

class HierarchicalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_options, action_dim):
        super(HierarchicalPolicy, self).__init__()
        self.option_policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        self.action_policy = nn.Sequential(
            nn.Linear(input_dim + num_options, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        option_logits = self.option_policy(state)
        option = torch.argmax(option_logits, dim=-1)
        option_onehot = torch.nn.functional.one_hot(option, num_classes=option_logits.size(-1))
        action_input = torch.cat([state, option_onehot], dim=-1)
        action_mean = self.action_policy(action_input)
        return option_logits, action_mean
