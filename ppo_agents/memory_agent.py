import numpy as np
from .base_agent import BasePPOAgent
from typing import List, Dict, Any
from drl_agents.models import Page

class MemoryManagerAgent(BasePPOAgent):
    def __init__(self, device='cpu'):
        # State: [page_faults, fault_rate, working_set_size, memory_pressure, access_frequency]
        state_size = 5
        # Actions: [LRU, FIFO, Clock, Random]
        action_size = 4
        super().__init__(state_size, action_size, device)
        
        self.access_history = []
        self.max_history = 1000
        self.access_frequency = {}
        
    def get_state(self, pages: List[Page], current_time: int) -> np.ndarray:
        if not pages:
            return np.zeros(self.state_size)
            
        # Calculate metrics
        page_faults = sum(1 for p in pages if p.is_fault)
        fault_rate = page_faults / len(pages)
        working_set = len(set(self.access_history[-100:]))  # Last 100 accesses
        memory_pressure = len(pages) / 1000  # Normalize to [0,1]
        
        # Calculate access frequency score
        if self.access_history:
            recent_accesses = self.access_history[-100:]
            freq_scores = [self.access_frequency.get(p.page_id, 0) / max(self.access_frequency.values())
                          for p in pages]
            avg_freq_score = np.mean(freq_scores)
        else:
            avg_freq_score = 0
            
        return np.array([
            page_faults / 1000,  # Normalize
            fault_rate,
            working_set / 100,
            memory_pressure,
            avg_freq_score
        ])
        
    def calculate_reward(self, metrics: Dict[str, float]) -> float:
        # Multi-objective reward
        fault_weight = 0.4
        latency_weight = 0.3
        efficiency_weight = 0.3
        
        # Normalize metrics
        fault_score = 1.0 - metrics['fault_rate']
        latency_score = 1.0 - (metrics['working_set_size'] / 100)
        efficiency_score = metrics.get('hit_rate', 0.0)
        
        return (fault_weight * fault_score +
                latency_weight * latency_score +
                efficiency_weight * efficiency_score)
        
    def optimize(self, pages: List[Page], access_pattern: List[int], current_time: int) -> Dict[str, Any]:
        # Update access history and frequency
        self.access_history.extend(access_pattern)
        if len(self.access_history) > self.max_history:
            self.access_history = self.access_history[-self.max_history:]
            
        for page_id in access_pattern:
            self.access_frequency[page_id] = self.access_frequency.get(page_id, 0) + 1
            
        state = self.get_state(pages, current_time)
        action, value, log_prob = self.choose_action(state)
        
        # Convert action to page replacement strategy
        if action == 0:  # LRU
            strategy = "LRU"
            victim = min(pages, key=lambda x: x.last_access).page_id
        elif action == 1:  # FIFO
            strategy = "FIFO"
            victim = min(pages, key=lambda x: x.loaded_time).page_id
        elif action == 2:  # Clock
            strategy = "Clock"
            victim = self._clock_algorithm(pages)
        else:  # Random
            strategy = "Random"
            victim = np.random.choice([p.page_id for p in pages])
            
        # Calculate metrics
        hits = len([p for p in pages if not p.is_fault])
        total_accesses = len(pages)
        
        metrics = {
            "page_faults": sum(1 for p in pages if p.is_fault),
            "fault_rate": len([p for p in pages if p.is_fault]) / len(pages),
            "working_set_size": len(set(self.access_history[-100:])),
            "hit_rate": hits / total_accesses if total_accesses > 0 else 0
        }
        
        # Calculate reward and store experience
        reward = self.calculate_reward(metrics)
        next_state = self.get_state(pages, current_time + 1)
        done = len(access_pattern) == 0
        
        # Store experience in memory
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.rewards.append(reward)
        self.memory.values.append(value)
        self.memory.logprobs.append(log_prob)
        self.memory.dones.append(done)
        
        # Learn if enough experiences are collected
        self.learn()
        
        return {
            "victim_page": victim,
            "metrics": metrics,
            "strategy": strategy
        }
        
    def _clock_algorithm(self, pages: List[Page]) -> int:
        if not hasattr(self, '_clock_hand'):
            self._clock_hand = 0
            
        while True:
            page = pages[self._clock_hand]
            if not page.referenced:
                victim = page.page_id
                self._clock_hand = (self._clock_hand + 1) % len(pages)
                return victim
            page.referenced = False
            self._clock_hand = (self._clock_hand + 1) % len(pages)
