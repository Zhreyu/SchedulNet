import numpy as np
from .base_agent import BasePPOAgent
from typing import List, Dict, Any
from drl_agents.models import DiskRequest

class DiskSchedulerAgent(BasePPOAgent):
    def __init__(self, device='cpu'):
        # State: [queue_length, avg_seek_distance, locality_score, direction_bias, request_density]
        state_size = 5
        # Actions: [FCFS, SSTF, SCAN, LOOK]
        action_size = 4
        super().__init__(state_size, action_size, device)
        
        self.current_track = 0
        self.direction = 1  # 1 for moving toward higher tracks, -1 for lower
        self.track_history = []
        self.max_history = 1000
        
    def get_state(self, requests: List[DiskRequest], current_time: int) -> np.ndarray:
        if not requests:
            return np.zeros(self.state_size)
            
        queue_length = len(requests)
        seek_distances = [abs(r.track - self.current_track) for r in requests]
        avg_seek_distance = np.mean(seek_distances)
        
        # Calculate locality score (how clustered the requests are)
        tracks = [r.track for r in requests]
        track_range = max(tracks) - min(tracks) if tracks else 0
        locality_score = 1.0 - (track_range / 1000)  # Normalize to [0,1]
        
        # Calculate direction bias (preference for current direction)
        direction_requests = sum(1 for r in requests if (r.track - self.current_track) * self.direction > 0)
        direction_bias = direction_requests / len(requests) if requests else 0
        
        # Calculate request density (concentration of requests in current region)
        current_region = (self.current_track // 100) * 100  # 100-track regions
        region_requests = sum(1 for r in requests if current_region <= r.track < current_region + 100)
        request_density = region_requests / len(requests) if requests else 0
        
        return np.array([
            queue_length / 100,  # Normalize
            avg_seek_distance / 1000,
            locality_score,
            direction_bias,
            request_density
        ])
        
    def calculate_reward(self, metrics: Dict[str, float], requests: List[DiskRequest]) -> float:
        # Multi-objective reward
        seek_weight = 0.4
        throughput_weight = 0.3
        fairness_weight = 0.3
        
        # Normalize metrics
        normalized_seek = 1.0 - min(metrics['avg_seek_time'] / 100, 1.0)
        normalized_throughput = min(metrics['total_seek_time'] / 10000, 1.0)
        
        # Calculate fairness (variance in response times)
        if requests:
            response_times = np.array([abs(r.track - self.current_track) for r in requests])
            variance = np.var(response_times) if len(response_times) > 1 else 0
            fairness = 1.0 - min(variance / 10000, 1.0)
        else:
            fairness = 1.0
            
        return (seek_weight * normalized_seek +
                throughput_weight * normalized_throughput +
                fairness_weight * fairness)
        
    def optimize(self, requests: List[DiskRequest], current_time: int) -> Dict[str, Any]:
        if not requests:
            return {
                "schedule": [],
                "metrics": {
                    "avg_seek_time": 0,
                    "total_seek_time": 0
                },
                "strategy": "None"
            }
            
        state = self.get_state(requests, current_time)
        action, value, log_prob = self.choose_action(state)
        
        # Convert action to scheduling strategy
        if action == 0:  # FCFS
            sorted_requests = sorted(requests, key=lambda x: x.arrival_time)
            strategy = "FCFS"
        elif action == 1:  # SSTF
            sorted_requests = sorted(requests, key=lambda x: abs(x.track - self.current_track))
            strategy = "SSTF"
        elif action == 2:  # SCAN
            sorted_requests = self._scan_algorithm(requests)
            strategy = "SCAN"
        else:  # LOOK
            sorted_requests = self._look_algorithm(requests)
            strategy = "LOOK"
            
        # Calculate metrics
        total_seek_time = 0
        current_track = self.current_track
        for req in sorted_requests:
            total_seek_time += abs(req.track - current_track)
            current_track = req.track
            
        metrics = {
            "avg_seek_time": total_seek_time / len(requests),
            "total_seek_time": total_seek_time
        }
        
        # Update current track and direction
        self.current_track = sorted_requests[-1].track if sorted_requests else self.current_track
        
        # Update track history
        self.track_history.append(self.current_track)
        if len(self.track_history) > self.max_history:
            self.track_history = self.track_history[-self.max_history:]
            
        # Calculate reward and store experience
        reward = self.calculate_reward(metrics, requests)
        next_state = self.get_state(requests[1:], current_time + 1)
        done = len(requests) <= 1
        
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
            "schedule": [
                {
                    "track": req.track,
                    "arrival_time": req.arrival_time,
                    "size": req.size
                }
                for req in sorted_requests
            ],
            "metrics": metrics,
            "strategy": strategy
        }
        
    def _scan_algorithm(self, requests: List[DiskRequest]) -> List[DiskRequest]:
        forward = [r for r in requests if r.track >= self.current_track]
        backward = [r for r in requests if r.track < self.current_track]
        
        if self.direction > 0:
            return sorted(forward) + sorted(backward, reverse=True)
        else:
            return sorted(backward, reverse=True) + sorted(forward)
            
    def _look_algorithm(self, requests: List[DiskRequest]) -> List[DiskRequest]:
        forward = [r for r in requests if r.track >= self.current_track]
        backward = [r for r in requests if r.track < self.current_track]
        
        if self.direction > 0:
            if forward:
                return sorted(forward)
            self.direction = -1
            return sorted(backward, reverse=True)
        else:
            if backward:
                return sorted(backward, reverse=True)
            self.direction = 1
            return sorted(forward)
