import numpy as np
from .base_agent import BaseRLAgent
from typing import List, Dict, Any

class DiskSchedulerAgent(BaseRLAgent):
    def __init__(self, device='cpu'):
        # State: [queue_length, avg_seek_distance, locality_score, direction_bias]
        state_size = 4
        # Actions: [FCFS, SSTF, SCAN, LOOK]
        action_size = 4
        super().__init__(state_size, action_size, device)
        
        self.current_track = 0
        self.direction = 1  # 1 for moving toward higher tracks, -1 for lower
        
    def get_state(self, requests: List[Any], current_time: int) -> np.ndarray:
        if not requests:
            return np.zeros(self.state_size)
            
        queue_length = len(requests)
        seek_distances = [abs(r.track - self.current_track) for r in requests]
        avg_seek_distance = np.mean(seek_distances)
        
        # Calculate locality score (how clustered the requests are)
        tracks = [r.track for r in requests]
        locality_score = 1.0 - (max(tracks) - min(tracks)) / 1000  # Normalize to [0,1]
        
        # Calculate direction bias (preference for current direction)
        direction_requests = sum(1 for r in requests if (r.track - self.current_track) * self.direction > 0)
        direction_bias = direction_requests / len(requests)
        
        return np.array([
            queue_length,
            avg_seek_distance,
            locality_score,
            direction_bias
        ])
        
    def calculate_reward(self, metrics: Dict[str, float]) -> float:
        # Reward based on seek time reduction
        seek_weight = 0.7
        throughput_weight = 0.3
        
        normalized_seek = 1.0 - min(metrics['avg_seek_time'] / 100, 1.0)  # Lower is better
        normalized_throughput = min(metrics['total_seek_time'] / 10000, 1.0)
        
        return (seek_weight * normalized_seek + 
                throughput_weight * normalized_throughput)
        
    def optimize(self, requests: List[Any], current_time: int) -> Dict[str, Any]:
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
        action = self.act(state)
        
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
        
        # Calculate reward and store experience
        reward = self.calculate_reward(metrics)
        next_state = self.get_state(requests[1:], current_time + 1)
        done = len(requests) <= 1
        
        self.remember(state, action, reward, next_state, done)
        self.replay()
        
        if done:
            self.update_target_net()
            
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
        
    def _scan_algorithm(self, requests: List[Any]) -> List[Any]:
        # Sort requests based on SCAN algorithm (elevator)
        forward = [r for r in requests if r.track >= self.current_track]
        backward = [r for r in requests if r.track < self.current_track]
        
        if self.direction > 0:
            return sorted(forward) + sorted(backward, reverse=True)
        else:
            return sorted(backward, reverse=True) + sorted(forward)
            
    def _look_algorithm(self, requests: List[Any]) -> List[Any]:
        # Similar to SCAN but only reverses direction when there are no more requests
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
