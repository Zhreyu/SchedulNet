"""Reinforcement Learning based memory management implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class MemoryDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(MemoryDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class RLMemoryManager:
    def __init__(self, num_frames, state_size=6):
        self.num_frames = num_frames
        self.frames = []
        self.page_faults = 0
        self.state_size = state_size
        self.action_size = num_frames  # Can replace any frame
        
        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9
        self.learning_rate = 0.001
        self.tau = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = MemoryDQN(state_size, self.action_size).to(self.device)
        self.target_model = MemoryDQN(state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.HuberLoss()
        
        # Initialize target network
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Statistics for state representation
        self.access_frequency = {}
        self.last_access = {}
        self.page_sizes = {}
        self.working_set = deque(maxlen=5)
        self.access_history = deque(maxlen=1000)
        
    def get_state(self, page):
        """Create state representation for current memory state."""
        current_time = len(self.memory)
        self.access_history.append(page)
        
        # Update statistics
        self.access_frequency[page] = self.access_frequency.get(page, 0) + 1
        self.last_access[page] = current_time
        self.working_set.append(page)
        
        if len(self.frames) < self.num_frames:
            return np.zeros(self.state_size)
            
        # Enhanced state features
        state = []
        for frame in self.frames:
            # 1. Access frequency (normalized)
            freq = self.access_frequency.get(frame, 0)
            max_freq = max(self.access_frequency.values()) if self.access_frequency else 1
            norm_freq = freq / max_freq
            
            # 2. Recency of access (normalized)
            recency = (current_time - self.last_access.get(frame, 0))
            max_recency = current_time if current_time > 0 else 1
            norm_recency = recency / max_recency
            
            # 3. Page size (normalized)
            size = self.page_sizes.get(frame, 0)
            max_size = max(self.page_sizes.values()) if self.page_sizes else 1
            norm_size = size / max_size
            
            # 4. Working set membership
            in_working_set = 1.0 if frame in self.working_set else 0.0
            
            # 5. Future reference probability (based on recent history)
            recent_refs = list(self.access_history)[-100:]
            future_ref_prob = sum(1 for ref in recent_refs if ref == frame) / len(recent_refs) if recent_refs else 0
            
            # 6. Page fault rate
            fault_rate = self.page_faults / (len(self.memory) + 1)
            
            state.extend([
                norm_freq,
                norm_recency,
                norm_size,
                in_working_set,
                future_ref_prob,
                fault_rate
            ])
            
        return np.array(state[-self.state_size:])
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
            
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            act_values = self.model(state)
            return torch.argmax(act_values).item()
            
    def replay(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
            
        # Sample and prepare batch
        minibatch = random.sample(self.memory, batch_size)
        
        # Convert to numpy arrays first
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values using target network
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Compute loss and update
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._update_target_model()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def _update_target_model(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
    def access_page(self, page, size=1):
        """Access a page and return reward based on hit/miss."""
        self.page_sizes[page] = size
        state = self.get_state(page)
        
        if page in self.frames:
            # Page hit: positive reward
            reward = 1.0
            done = False
            action = self.frames.index(page)
        else:
            # Page fault: negative reward
            self.page_faults += 1
            reward = -1.0
            
            if len(self.frames) >= self.num_frames:
                # Need to replace a page
                action = self.act(state)
                if action < len(self.frames):
                    self.frames[action] = page
            else:
                self.frames.append(page)
                action = len(self.frames) - 1
                
            done = len(self.frames) >= self.num_frames
            
        next_state = self.get_state(page)
        self.remember(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            self.replay()
            
        return self.page_faults
