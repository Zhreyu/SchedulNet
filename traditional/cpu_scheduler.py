"""Traditional CPU scheduling algorithms."""

from typing import List
import numpy as np
from benchmarks.workload_generator import WorkloadTask

class FCFSScheduler:
    """First Come First Serve Scheduler."""
    
    def schedule(self, tasks: List[WorkloadTask]) -> List[float]:
        """Schedule tasks using FCFS."""
        current_time = 0
        completion_times = []
        
        # Sort by arrival time
        sorted_tasks = sorted(tasks, key=lambda x: x.arrival_time)
        
        for task in sorted_tasks:
            # Wait until task arrives if necessary
            current_time = max(current_time, task.arrival_time)
            # Execute task
            current_time += task.burst_time
            completion_times.append(current_time)
            
        return completion_times

class SJFScheduler:
    """Shortest Job First Scheduler."""
    
    def schedule(self, tasks: List[WorkloadTask]) -> List[float]:
        """Schedule tasks using SJF."""
        current_time = 0
        completion_times = [0] * len(tasks)
        remaining_tasks = list(enumerate(tasks))  # Keep track of original indices
        
        while remaining_tasks:
            # Get available tasks
            available = [
                (i, task) for i, task in remaining_tasks
                if task.arrival_time <= current_time
            ]
            
            if not available:
                # Jump to next arrival time
                current_time = min(task.arrival_time for _, task in remaining_tasks)
                continue
                
            # Choose shortest job
            next_task_idx, next_task = min(available, key=lambda x: x[1].burst_time)
            
            # Execute task
            current_time += next_task.burst_time
            completion_times[next_task_idx] = current_time
            
            # Remove from remaining
            remaining_tasks.remove((next_task_idx, next_task))
            
        return completion_times

class RRScheduler:
    """Round Robin Scheduler."""
    
    def __init__(self, quantum: int = 2):
        self.quantum = quantum
        
    def schedule(self, tasks: List[WorkloadTask]) -> List[float]:
        """Schedule tasks using Round Robin."""
        current_time = 0
        completion_times = [0] * len(tasks)
        remaining_bursts = [(i, task.burst_time) for i, task in enumerate(tasks)]
        
        while remaining_bursts:
            executed = False
            
            for i, (task_idx, burst) in enumerate(remaining_bursts):
                if tasks[task_idx].arrival_time <= current_time:
                    # Execute for quantum or remaining time
                    execution_time = min(self.quantum, burst)
                    current_time += execution_time
                    
                    # Update remaining burst
                    remaining_bursts[i] = (task_idx, burst - execution_time)
                    
                    if remaining_bursts[i][1] == 0:
                        completion_times[task_idx] = current_time
                        remaining_bursts.pop(i)
                        
                    executed = True
                    break
                    
            if not executed:
                # Jump to next arrival time
                current_time = min(
                    tasks[idx].arrival_time 
                    for idx, _ in remaining_bursts
                )
                
        return completion_times
