"""Traditional CPU scheduling algorithms implementation."""

from dataclasses import dataclass
from typing import List, Optional
import heapq

@dataclass
class Process:
    pid: int
    arrival_time: int
    burst_time: int
    priority: Optional[int] = None
    remaining_time: Optional[int] = None
    completed: bool = False
    
    def __post_init__(self):
        if self.remaining_time is None:
            self.remaining_time = self.burst_time
            
    def __lt__(self, other):
        return self.burst_time < other.burst_time
    
class CPUScheduler:
    def __init__(self):
        self.ready_queue = []
        self.current_time = 0
        self.completed_processes = []
    
    def schedule(self, processes: List[Process]):
        raise NotImplementedError

class RoundRobinScheduler(CPUScheduler):
    def __init__(self, time_quantum):
        super().__init__()
        self.time_quantum = time_quantum
    
    def schedule(self, processes: List[Process]):
        remaining_processes = processes.copy()
        while remaining_processes or self.ready_queue:
            # Add newly arrived processes to ready queue
            while remaining_processes and remaining_processes[0].arrival_time <= self.current_time:
                self.ready_queue.append(remaining_processes.pop(0))
            
            if not self.ready_queue:
                self.current_time = remaining_processes[0].arrival_time
                continue
            
            current_process = self.ready_queue.pop(0)
            execution_time = min(self.time_quantum, current_process.remaining_time)
            self.current_time += execution_time
            current_process.remaining_time -= execution_time
            
            if current_process.remaining_time > 0:
                self.ready_queue.append(current_process)
            else:
                current_process.completed = True
                self.completed_processes.append(current_process)

class SJFScheduler(CPUScheduler):
    def schedule(self, processes: List[Process]):
        remaining_processes = [(p.burst_time, p) for p in processes]
        heapq.heapify(remaining_processes)
        
        while remaining_processes:
            burst_time, process = heapq.heappop(remaining_processes)
            self.current_time += burst_time
            process.completed = True
            self.completed_processes.append(process)
