"""LLM agent for disk scheduling."""

import json
from typing import List, Dict, Any
from openai import OpenAI
from traditional.disk_scheduling import DiskRequest

class DiskSchedulerAgent:
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        self.prompt = """You are a disk scheduling agent.
        Your task is to decide how to schedule disk I/O requests.
        Respond with ONLY a single character:
        'F' - First Come First Served
        'S' - Shortest Seek Time First
        'E' - Elevator (SCAN)
        'L' - Look
        """
        
    def optimize(self, requests: List[DiskRequest]) -> Dict[str, Any]:
        """Optimize disk scheduling for given requests."""
        if not requests:
            return {
                "schedule": [],
                "metrics": {
                    "avg_seek_time": 0,
                    "total_seek_time": 0
                },
                "reason": "No requests to schedule"
            }
            
        # Prepare request data for LLM
        request_data = [
            {
                "track": req.track,
                "arrival_time": req.arrival_time,
                "size": req.size,
                "priority": req.priority
            }
            for req in requests
        ]
        
        prompt = f"""Current disk queue:
        {json.dumps(request_data)}
        Which disk scheduling algorithm should be used? Respond with a single character.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                stream=False
            )
            
            decision = response.choices[0].message.content.strip()
            
            # Convert single character decision to scheduling strategy
            if decision == 'S':
                # SSTF: sort by track distance from current position
                current_track = requests[0].track
                sorted_requests = sorted(requests, key=lambda x: abs(x.track - current_track))
                strategy = "SSTF"
            elif decision == 'E':
                # SCAN: sort by track number in current direction
                sorted_requests = sorted(requests, key=lambda x: x.track)
                strategy = "SCAN"
            elif decision == 'L':
                # LOOK: like SCAN but only go as far as needed
                sorted_requests = sorted(requests, key=lambda x: (x.track, x.arrival_time))
                strategy = "LOOK"
            else:  # 'F' or fallback
                # FCFS: sort by arrival time
                sorted_requests = sorted(requests, key=lambda x: x.arrival_time)
                strategy = "FCFS"
            
            # Calculate metrics
            total_seek_time = 0
            current_track = sorted_requests[0].track
            for req in sorted_requests[1:]:
                total_seek_time += abs(req.track - current_track)
                current_track = req.track
            
            return {
                "schedule": [
                    {
                        "track": req.track,
                        "arrival_time": req.arrival_time,
                        "size": req.size
                    }
                    for req in sorted_requests
                ],
                "metrics": {
                    "avg_seek_time": total_seek_time / len(requests),
                    "total_seek_time": total_seek_time
                },
                "strategy": strategy
            }
            
        except Exception as e:
            print(f"Error in disk scheduling: {e}")
            # Fallback to FCFS
            sorted_requests = sorted(requests, key=lambda x: x.arrival_time)
            return {
                "schedule": [
                    {
                        "track": req.track,
                        "arrival_time": req.arrival_time,
                        "size": req.size
                    }
                    for req in sorted_requests
                ],
                "metrics": {
                    "avg_seek_time": float('inf'),
                    "total_seek_time": float('inf')
                },
                "strategy": "FCFS (fallback)"
            }
