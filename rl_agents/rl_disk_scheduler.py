"""Reinforcement Learning based disk scheduling implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DiskDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DiskDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLDiskScheduler:
    def __init__(self, total_tracks=200, state_size=6):
        self.total_tracks = total_tracks
        self.current_track = 0
        self.total_seek_time = 0
        self.direction = 1
        
        self.state_size = state_size
        self.action_size = 2  # 0: serve request, 1: skip to next
        
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DiskDQN(state_size, self.action_size).to(self.device)
        self.target_model = DiskDQN(state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Statistics for state representation
        self.request_frequency = {}
        self.last_service = {}
        self.request_sizes = {}
        
    def get_state(self, request, pending_requests):
        """Create state representation for current disk state."""
        current_time = len(self.memory)
        
        # Update statistics
        self.request_frequency[request.track] = self.request_frequency.get(request.track, 0) + 1
        self.last_service[request.track] = current_time
        self.request_sizes[request.track] = request.size
        
        # State features:
        # 1. Current track position (normalized)
        # 2. Request track position (normalized)
        # 3. Direction of head movement
        # 4. Request frequency
        # 5. Time since last service
        # 6. Request size
        
        state = [
            self.current_track / self.total_tracks,
            request.track / self.total_tracks,
            self.direction,
            self.request_frequency.get(request.track, 0) / max(self.request_frequency.values()),
            (current_time - self.last_service.get(request.track, 0)) / current_time if current_time > 0 else 0,
            request.size / max(self.request_sizes.values()) if self.request_sizes else 0
        ]
        
        return np.array(state)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return np.argmax(act_values.cpu().numpy())
            
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch]).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in minibatch]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def schedule(self, requests):
        """Schedule disk requests using RL."""
        seek_times = []
        pending_requests = requests.copy()
        
        while pending_requests:
            min_seek_time = float('inf')
            next_request = None
            next_index = 0
            
            for i, request in enumerate(pending_requests):
                state = self.get_state(request, pending_requests)
                action = self.act(state)
                
                if action == 0:  # Serve this request
                    seek_time = abs(self.current_track - request.track)
                    if seek_time < min_seek_time:
                        min_seek_time = seek_time
                        next_request = request
                        next_index = i
            
            if next_request is None:
                # If no request was selected, choose the nearest one
                for i, request in enumerate(pending_requests):
                    seek_time = abs(self.current_track - request.track)
                    if seek_time < min_seek_time:
                        min_seek_time = seek_time
                        next_request = request
                        next_index = i
            
            # Execute the selected request
            seek_time = abs(self.current_track - next_request.track)
            self.total_seek_time += seek_time
            seek_times.append(seek_time)
            
            # Update state and learn
            old_state = self.get_state(next_request, pending_requests)
            self.current_track = next_request.track
            pending_requests.pop(next_index)
            
            # Calculate reward (negative seek time, normalized)
            reward = -seek_time / self.total_tracks
            
            if pending_requests:
                next_state = self.get_state(pending_requests[0], pending_requests)
                done = False
            else:
                next_state = np.zeros(self.state_size)
                done = True
                
            self.remember(old_state, 0, reward, next_state, done)
            
            if len(self.memory) > 32:
                self.replay(32)
                
        return seek_times
