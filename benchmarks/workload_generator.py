"""Synthetic workload generator for benchmarking."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from traditional.disk_scheduling import DiskRequest

@dataclass
class WorkloadTask:
    task_id: int
    arrival_time: int
    burst_time: int
    memory_size: int  # in KB
    priority: Optional[int] = None
    waiting_time: int = 0
    completed: bool = False

class WorkloadGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def generate_cpu_workload(self, num_tasks: int, 
                             arrival_time_range: tuple = (0, 100),
                             burst_time_range: tuple = (1, 20),
                             memory_range: tuple = (64, 1024)) -> List[WorkloadTask]:
        """Generate a synthetic CPU workload with specified parameters."""
        tasks = []
        for i in range(num_tasks):
            arrival_time = self.rng.randint(*arrival_time_range)
            burst_time = self.rng.randint(*burst_time_range)
            memory_size = self.rng.randint(*memory_range)
            priority = self.rng.randint(1, 10)
            
            tasks.append(WorkloadTask(
                task_id=i,
                arrival_time=arrival_time,
                burst_time=burst_time,
                memory_size=memory_size,
                priority=priority
            ))
            
        # Sort by arrival time
        return sorted(tasks, key=lambda x: x.arrival_time)
    
    def generate_memory_workload(self, num_pages: int,
                               page_range: tuple = (0, 100),
                               page_size_range: tuple = (4, 64)) -> List[tuple]:
        """Generate a sequence of page references with sizes."""
        # Create localities of reference
        localities = self.rng.randint(*page_range, size=num_pages // 4)
        
        # Generate page references with temporal locality
        pages = []
        for _ in range(num_pages):
            if self.rng.random() < 0.7:  # 70% chance to reference from localities
                page = self.rng.choice(localities)
            else:
                page = self.rng.randint(*page_range)
                
            # Add page size (in KB)
            size = self.rng.randint(*page_size_range)
            pages.append((page, size))
                
        return pages
        
    def generate_disk_workload(self, num_requests: int,
                              track_range: tuple = (0, 199),
                              arrival_time_range: tuple = (0, 100),
                              request_size_range: tuple = (4, 128)) -> List[DiskRequest]:
        """Generate disk I/O requests with locality patterns."""
        requests = []
        
        # Create zones of high activity
        hot_zones = [
            (0, track_range[1] // 3),
            (track_range[1] // 3, 2 * track_range[1] // 3),
            (2 * track_range[1] // 3, track_range[1])
        ]
        
        for i in range(num_requests):
            if self.rng.random() < 0.8:  # 80% chance to access hot zones
                zone_start, zone_end = hot_zones[self.rng.randint(0, len(hot_zones))]
                track = self.rng.randint(zone_start, zone_end)
            else:
                track = self.rng.randint(*track_range)
                
            arrival_time = self.rng.randint(*arrival_time_range)
            request_size = self.rng.randint(*request_size_range)
            
            requests.append(DiskRequest(
                track=track,
                arrival_time=arrival_time,
                size=request_size,
                priority=self.rng.randint(1, 10)
            ))
            
        # Sort by arrival time
        return sorted(requests, key=lambda x: x.arrival_time)
