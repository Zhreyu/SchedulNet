import numpy as np
from .base_agent import BasePPOAgent
from typing import List, Dict, Any
from drl_agents.models import Process

class CPUSchedulerAgent(BasePPOAgent):
    def __init__(self, device='cpu'):
        # State: [queue_length, avg_waiting_time, avg_burst_time, avg_priority, cpu_utilization]
        state_size = 5
        # Actions: [FCFS, SJF, Priority, RR]
        action_size = 4
        super().__init__(state_size, action_size, device)
        
        self.total_time = 0
        self.busy_time = 0
        
    def get_state(self, workload: List[Process], current_time: int) -> np.ndarray:
        if not workload:
            return np.zeros(self.state_size)
            
        queue_length = len(workload)
        avg_waiting_time = np.mean([current_time - p.arrival_time for p in workload])
        avg_burst_time = np.mean([p.burst_time for p in workload])
        avg_priority = np.mean([p.priority for p in workload])
        cpu_utilization = self.busy_time / max(self.total_time, 1)
        
        return np.array([
            queue_length / 100,  # Normalize
            avg_waiting_time / 100,
            avg_burst_time / 20,
            avg_priority / 10,
            cpu_utilization
        ])
        
    def calculate_reward(self, metrics: Dict[str, float], workload: List[Process]) -> float:
        # Multi-objective reward
        completion_weight = 0.4
        throughput_weight = 0.3
        fairness_weight = 0.3
        
        # Normalize metrics
        normalized_completion = 1.0 - min(metrics['avg_completion_time'] / 100, 1.0)
        normalized_throughput = min(metrics['throughput'] / 1000000, 1.0)
        
        # Calculate fairness (Jain's fairness index)
        waiting_times = np.array([metrics['avg_completion_time'] for _ in workload])
        fairness = np.square(np.sum(waiting_times)) / (len(waiting_times) * np.sum(np.square(waiting_times)))
        
        return (completion_weight * normalized_completion +
                throughput_weight * normalized_throughput +
                fairness_weight * fairness)
        
    def optimize(self, workload: List[Process], current_time: int) -> Dict[str, Any]:
        state = self.get_state(workload, current_time)
        action, value, log_prob = self.choose_action(state)
        
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
            next_process = workload[0]
            strategy = "RR"
            
        # Update CPU utilization tracking
        self.total_time += 1
        if workload:
            self.busy_time += 1
            
        # Calculate metrics
        completed_processes = [p for p in workload if p.completed]
        waiting_time = current_time - next_process.arrival_time if hasattr(next_process, 'arrival_time') else 0
        
        metrics = {
            "avg_completion_time": waiting_time,
            "max_completion_time": max((current_time - p.arrival_time for p in completed_processes), default=0),
            "throughput": len(completed_processes) / (current_time + 1) if current_time > 0 else 0
        }
        
        # Calculate reward and store experience
        reward = self.calculate_reward(metrics, workload)
        next_state = self.get_state(workload[1:], current_time + 1)
        done = len(workload) <= 1
        
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
            "next_process": next_process.pid,
            "time_slice": min(next_process.burst_time, 10),  # Cap at 10 time units
            "metrics": metrics,
            "strategy": strategy
        }
