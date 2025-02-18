import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, List

class ActorCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        return self.actor(features), self.critic(features)

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.dones = []
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.logprobs.clear()
        self.dones.clear()
        
    def compute_advantages(self, gamma: float, gae_lambda: float) -> torch.Tensor:
        rewards = torch.tensor(self.rewards)
        values = torch.tensor(self.values)
        dones = torch.tensor(self.dones)
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

class BasePPOAgent:
    def __init__(self, state_size: int, action_size: int, device: str = 'cpu'):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.c1 = 1.0  # Value loss coefficient
        self.c2 = 0.01  # Entropy coefficient
        self.batch_size = 64
        self.n_epochs = 10
        self.learning_rate = 3e-4
        
        # Neural network and optimizer
        self.ac_net = ActorCritic(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=self.learning_rate)
        
        # Memory buffer
        self.memory = PPOMemory()
        
    def choose_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs, value = self.ac_net(state)
            
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), value.item(), log_prob.item()
        
    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return
            
        # Convert memory to tensors
        states = torch.FloatTensor(self.memory.states).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.memory.compute_advantages(self.gamma, self.gae_lambda)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # PPO update for n_epochs
        for _ in range(self.n_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                
                # Forward pass
                probs, values = self.ac_net(batch_states)
                dist = Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratios and surrogate losses
                ratios = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                
                # Calculate losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
                self.optimizer.step()
                
        # Clear memory after update
        self.memory.clear()
        
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.ac_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.ac_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
