import torch
import numpy as np
from environment.simple_env import make_env
from agents.hierarchical_agent import HierarchicalAgent
from utils.replay_buffer import ReplayBuffer

# GPU 사용 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def state_to_graph(observation):
    # Waterworld의 관찰 공간에 맞게 수정
    x = torch.tensor(observation, dtype=torch.float32).unsqueeze(1).to(device)  # (162, 1)
    num_nodes = x.shape[0]
    # 완전 연결 그래프 생성
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], 
                              dtype=torch.long).t().contiguous().to(device)
    return x, edge_index

def train(env, agents, num_episodes, batch_size):
    replay_buffers = {agent: ReplayBuffer(10000) for agent in env.agents}
    
    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.agents}
        
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            total_rewards[agent] += reward

            if termination or truncation:
                action = None
            else:
                state, edge_index = state_to_graph(observation)
                action = agents[agent].act(state, edge_index)
                
                if len(replay_buffers[agent]) > batch_size:
                    batch = replay_buffers[agent].sample(batch_size)
                    loss = agents[agent].update(batch)
            
            if action is not None:
                action = action.clip(-1, 1)  # 액션을 [-1, 1] 범위로 클리핑
            env.step(action)
            
            if not (termination or truncation):
                next_observation, next_reward, next_termination, next_truncation, next_info = env.last()
                next_state, next_edge_index = state_to_graph(next_observation)
                replay_buffers[agent].add(state.cpu().numpy(), edge_index.cpu().numpy(), action, reward, next_state.cpu().numpy(), next_edge_index.cpu().numpy(), termination or truncation)

        print(f"Episode {episode + 1}, Total Rewards: {total_rewards}")

def main():
    # Hyperparameters
    num_episodes = 1000
    batch_size = 32
    hidden_dim = 64
    num_options = 4
    
    # Initialize environment and agents
    env = make_env()
    env.reset()
    state_dim = env.observation_space(env.agents[0]).shape[0]
    action_dim = env.action_space(env.agents[0]).shape[0]
    
    agents = {agent: HierarchicalAgent(state_dim, action_dim, hidden_dim, num_options, device) for agent in env.agents}
    
    # Train the agents
    train(env, agents, num_episodes, batch_size)

if __name__ == "__main__":
    main()
