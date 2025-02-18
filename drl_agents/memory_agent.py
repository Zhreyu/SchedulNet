import numpy as np
from .base_agent import BaseRLAgent
from typing import List, Dict, Any

class MemoryManagerAgent(BaseRLAgent):
    def __init__(self, device='cpu'):
        # State: [page_faults, fault_rate, working_set_size, memory_pressure]
        state_size = 4
        # Actions: [LRU, FIFO, Clock, Random]
        action_size = 4
        super().__init__(state_size, action_size, device)
        
        self.access_history = []
        self.max_history = 1000
        
    def get_state(self, pages: List[Any], current_time: int) -> np.ndarray:
        if not pages:
            return np.zeros(self.state_size)
            
        # Calculate metrics
        page_faults = sum(1 for p in pages if p.is_fault)
        fault_rate = page_faults / len(pages)
        working_set = len(set(self.access_history[-100:]))  # Last 100 accesses
        memory_pressure = len(pages) / 1000  # Normalize to [0,1]
        
        return np.array([
            page_faults,
            fault_rate,
            working_set,
            memory_pressure
        ])
        
    def calculate_reward(self, metrics: Dict[str, float]) -> float:
        # Reward based on fault rate (lower is better)
        return 1.0 - min(metrics['fault_rate'], 1.0)
        
    def optimize(self, pages: List[Any], access_pattern: List[int], current_time: int) -> Dict[str, Any]:
        # Update access history
        self.access_history.extend(access_pattern)
        if len(self.access_history) > self.max_history:
            self.access_history = self.access_history[-self.max_history:]
            
        state = self.get_state(pages, current_time)
        action = self.act(state)
        
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
        metrics = {
            "page_faults": sum(1 for p in pages if p.is_fault),
            "fault_rate": len([p for p in pages if p.is_fault]) / len(pages),
            "working_set_size": len(set(self.access_history[-100:]))
        }
        
        # Calculate reward and store experience
        reward = self.calculate_reward(metrics)
        next_state = self.get_state(pages, current_time + 1)
        done = len(access_pattern) == 0
        
        self.remember(state, action, reward, next_state, done)
        self.replay()
        
        if done:
            self.update_target_net()
            
        return {
            "victim_page": victim,
            "metrics": metrics,
            "strategy": strategy
        }
        
    def _clock_algorithm(self, pages: List[Any]) -> int:
        # Simple clock algorithm implementation
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
