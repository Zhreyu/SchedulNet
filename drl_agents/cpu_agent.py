import numpy as np
from .base_agent import BaseRLAgent
from typing import List, Dict, Any

class CPUSchedulerAgent(BaseRLAgent):
    def __init__(self, device='cpu'):
        # State: [queue_length, avg_waiting_time, avg_burst_time, avg_priority]
        state_size = 4
        # Actions: [FCFS, SJF, Priority, RR]
        action_size = 4
        super().__init__(state_size, action_size, device)
        
    def get_state(self, workload: List[Any], current_time: int) -> np.ndarray:
        if not workload:
            return np.zeros(self.state_size)
            
        queue_length = len(workload)
        avg_waiting_time = np.mean([current_time - p.arrival_time for p in workload])
        avg_burst_time = np.mean([p.burst_time for p in workload])
        avg_priority = np.mean([p.priority for p in workload])
        
        return np.array([
            queue_length,
            avg_waiting_time,
            avg_burst_time,
            avg_priority
        ])
        
    def calculate_reward(self, metrics: Dict[str, float]) -> float:
        # Reward based on throughput and completion time
        throughput_weight = 0.6
        completion_weight = 0.4
        
        normalized_throughput = min(metrics['throughput'] / 1000000, 1.0)  # Normalize to [0,1]
        normalized_completion = 1.0 - min(metrics['avg_completion_time'] / 100, 1.0)  # Lower is better
        
        return (throughput_weight * normalized_throughput + 
                completion_weight * normalized_completion)
        
    def optimize(self, workload: List[Any], current_time: int) -> Dict[str, Any]:
        state = self.get_state(workload, current_time)
        action = self.act(state)
        
        # Convert action to scheduling decision
        if action == 0:  # FCFS
            next_process = min(workload, key=lambda x: x.arrival_time)
            strategy = "FCFS"
        elif action == 1:  # SJF
            next_process = min(workload, key=lambda x: x.burst_time)
            strategy = "SJF"
        elif action == 2:  # Priority
            next_process = max(workload, key=lambda x: x.priority)
            strategy = "Priority"
        else:  # Round Robin
            next_process = workload[0]  # Take first process
            strategy = "RR"
            
        # Calculate metrics
        completed_processes = [p for p in workload if hasattr(p, 'completed') and p.completed]
        waiting_time = current_time - next_process.arrival_time if hasattr(next_process, 'arrival_time') else 0
        
        metrics = {
            "avg_completion_time": waiting_time,
            "max_completion_time": max((current_time - p.arrival_time for p in completed_processes), default=0),
            "throughput": len(completed_processes) / (current_time + 1) if current_time > 0 else 0
        }
        
        # Calculate reward and store experience
        reward = self.calculate_reward(metrics)
        next_state = self.get_state(workload[1:], current_time + 1)
        done = len(workload) <= 1
        
        self.remember(state, action, reward, next_state, done)
        self.replay()
        
        if done:
            self.update_target_net()
            
        return {
            "next_process": next_process.pid,
            "time_slice": min(next_process.burst_time, 10),  # Cap at 10 time units
            "metrics": metrics,
            "strategy": strategy
        }
