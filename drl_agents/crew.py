import numpy as np
from .base_agent import BaseRLAgent
from typing import Dict, Any

class CrewCoordinator(BaseRLAgent):
    def __init__(self, device='cpu'):
        # State: [cpu_performance, memory_performance, disk_performance, system_load]
        state_size = 4
        # Actions: [Accept, Modify CPU, Modify Memory, Modify Disk]
        action_size = 4
        super().__init__(state_size, action_size, device)
        
    def get_state(self, cpu_metrics: Dict[str, float],
                 memory_metrics: Dict[str, float],
                 disk_metrics: Dict[str, float]) -> np.ndarray:
        # Normalize metrics to [0,1] range
        cpu_perf = min(cpu_metrics.get('throughput', 0) / 1000000, 1.0)
        memory_perf = 1.0 - min(memory_metrics.get('fault_rate', 1.0), 1.0)
        disk_perf = 1.0 - min(disk_metrics.get('avg_seek_time', 100) / 100, 1.0)
        
        # Calculate system load (simple average of component stress)
        system_load = (cpu_perf + memory_perf + disk_perf) / 3
        
        return np.array([
            cpu_perf,
            memory_perf,
            disk_perf,
            system_load
        ])
        
    def calculate_reward(self, state: np.ndarray) -> float:
        # Reward based on overall system performance
        component_weights = [0.4, 0.3, 0.2, 0.1]  # CPU, Memory, Disk, Load
        return np.sum(state * component_weights)
        
    def coordinate_decisions(self, 
                           cpu_decision: Dict[str, Any],
                           memory_decision: Dict[str, Any],
                           disk_decision: Dict[str, Any]) -> Dict[str, Any]:
        
        state = self.get_state(
            cpu_decision.get('metrics', {}),
            memory_decision.get('metrics', {}),
            disk_decision.get('metrics', {})
        )
        
        action = self.act(state)
        
        # Calculate reward and store experience
        reward = self.calculate_reward(state)
        # For simplicity, next state is the same as current state
        next_state = state
        done = False
        
        self.remember(state, action, reward, next_state, done)
        self.replay()
        
        # Convert action to coordination decision
        if action == 0:  # Accept all
            return {
                "approved": True,
                "adjustments": {},
                "reasoning": "All decisions approved"
            }
        elif action == 1:  # Modify CPU
            return {
                "approved": False,
                "adjustments": {
                    "cpu": {"strategy": "RR"}  # Default to Round Robin
                },
                "reasoning": "CPU schedule needs adjustment"
            }
        elif action == 2:  # Modify Memory
            return {
                "approved": False,
                "adjustments": {
                    "memory": {"strategy": "LRU"}  # Default to LRU
                },
                "reasoning": "Memory management needs adjustment"
            }
        else:  # Modify Disk
            return {
                "approved": False,
                "adjustments": {
                    "disk": {"strategy": "SSTF"}  # Default to SSTF
                },
                "reasoning": "Disk schedule needs adjustment"
            }
