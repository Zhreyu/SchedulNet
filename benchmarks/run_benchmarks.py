"""Benchmark runner for comparing traditional and AI-based scheduling."""

import sys
import os
import time
import numpy as np
import torch
from typing import List, Dict
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traditional.cpu_scheduling import RoundRobinScheduler, SJFScheduler, Process
from traditional.memory_management import FIFOPageReplacement, LRUPageReplacement
from traditional.disk_scheduling import (
    FCFSDiskScheduler, SSTFDiskScheduler, SCANDiskScheduler,
    C_SCANDiskScheduler, LOOKDiskScheduler
)
from ai_agents.rl_scheduler import RLScheduler
from ai_agents.rl_memory_scheduler import RLMemoryManager
from ai_agents.rl_disk_scheduler import RLDiskScheduler
from benchmarks.workload_generator import WorkloadGenerator
from benchmarks.performance_metrics import PerformanceMonitor

class BenchmarkRunner:
    def __init__(self):
        self.workload_gen = WorkloadGenerator()
        self.perf_monitor = PerformanceMonitor()
        self.current_time = 0
        
    def _convert_workload_to_processes(self, workload) -> List[Process]:
        return [Process(
            pid=task.task_id,
            arrival_time=task.arrival_time,
            burst_time=task.burst_time,
            priority=task.priority
        ) for task in workload]
    
    def _train_rl_scheduler(self, workload, episodes=10):
        """Train the RL scheduler."""
        state_size = 4  # [waiting_time, burst_time, priority, queue_length]
        action_size = len(workload)
        scheduler = RLScheduler(state_size, action_size)
        current_time = 0
        
        for episode in range(episodes):
            # Reset episode state
            current_time = 0
            for task in workload:
                task.waiting_time = 0
                task.completed = False
                
            state = np.zeros(state_size)
            done = False
            total_reward = 0
            
            while not done:
                action = scheduler.act(state)
                task = workload[action]
                
                # Update waiting time
                task.waiting_time = max(0, current_time - task.arrival_time)
                current_time += task.burst_time
                task.completed = True
                
                # Create next state
                next_state = np.array([
                    task.waiting_time,
                    task.burst_time,
                    task.priority if task.priority else 0,
                    sum(1 for t in workload if not t.completed)
                ])
                
                # Negative waiting time as reward
                reward = -task.waiting_time
                scheduler.remember(state, action, reward, next_state, done)
                state = next_state
                
                if len(scheduler.memory) > scheduler.batch_size:
                    scheduler.replay()
                
                total_reward += reward
                if all(t.completed for t in workload):
                    done = True
            
            # Update target network less frequently
            if episode % 2 == 0:  
                scheduler._update_target_model()
            
        return scheduler
    
    def run_cpu_scheduling_benchmark(self, num_tasks=20):
        print("\nRunning CPU Scheduling Benchmark...")
        workload = self.workload_gen.generate_cpu_workload(num_tasks)
        processes = self._convert_workload_to_processes(workload)
        
        results = {}
        
        # Test Round Robin
        rr_scheduler = RoundRobinScheduler(time_quantum=4)
        start_time = time.time()
        self.perf_monitor.collect_metrics()
        rr_scheduler.schedule(processes.copy())
        results['round_robin'] = {
            'completion_time': time.time() - start_time,
            'metrics': self.perf_monitor.get_average_metrics()
        }
        
        # Test SJF
        sjf_scheduler = SJFScheduler()
        start_time = time.time()
        self.perf_monitor.collect_metrics()
        sjf_scheduler.schedule(processes.copy())
        results['sjf'] = {
            'completion_time': time.time() - start_time,
            'metrics': self.perf_monitor.get_average_metrics()
        }
        
        # Test RL Scheduler
        rl_scheduler = self._train_rl_scheduler(workload)
        start_time = time.time()
        self.perf_monitor.collect_metrics()
        # Use trained model for scheduling
        for _ in range(len(processes)):
            state = np.array([0, 0, 0, len(processes)])  # Simplified state
            action = rl_scheduler.act(state)
            # Execute action (in real implementation, this would schedule the task)
        results['rl_scheduler'] = {
            'completion_time': time.time() - start_time,
            'metrics': self.perf_monitor.get_average_metrics()
        }
        
        return results
    
    def run_memory_management_benchmark(self, num_pages=1000, num_frames=50):
        print("\nRunning Memory Management Benchmark...")
        page_references = self.workload_gen.generate_memory_workload(num_pages)
        
        results = {}
        
        # Test FIFO
        fifo = FIFOPageReplacement(num_frames)
        start_time = time.time()
        self.perf_monitor.collect_metrics()
        for page, size in page_references:
            fifo.access_page(page)
        results['fifo'] = {
            'completion_time': time.time() - start_time,
            'page_faults': fifo.page_faults,
            'metrics': self.perf_monitor.get_average_metrics()
        }
        
        # Test LRU
        lru = LRUPageReplacement(num_frames)
        start_time = time.time()
        self.perf_monitor.collect_metrics()
        for page, size in page_references:
            lru.access_page(page)
        results['lru'] = {
            'completion_time': time.time() - start_time,
            'page_faults': lru.page_faults,
            'metrics': self.perf_monitor.get_average_metrics()
        }
        
        # Test RL Memory Manager
        rl_memory = RLMemoryManager(num_frames)
        start_time = time.time()
        self.perf_monitor.collect_metrics()
        for page, size in page_references:
            rl_memory.access_page(page, size)
        results['rl_memory'] = {
            'completion_time': time.time() - start_time,
            'page_faults': rl_memory.page_faults,
            'metrics': self.perf_monitor.get_average_metrics()
        }
        
        return results
    
    def run_disk_scheduling_benchmark(self, num_requests=100):
        print("\nRunning Disk Scheduling Benchmark...")
        requests = self.workload_gen.generate_disk_workload(num_requests)
        
        results = {}
        schedulers = {
            'fcfs': FCFSDiskScheduler(),
            'sstf': SSTFDiskScheduler(),
            'scan': SCANDiskScheduler(),
            'c-scan': C_SCANDiskScheduler(),
            'look': LOOKDiskScheduler(),
            'rl': RLDiskScheduler()
        }
        
        for name, scheduler in schedulers.items():
            start_time = time.time()
            self.perf_monitor.collect_metrics()
            seek_times = scheduler.schedule(requests.copy())
            results[name] = {
                'completion_time': time.time() - start_time,
                'total_seek_time': sum(seek_times),
                'avg_seek_time': np.mean(seek_times),
                'max_seek_time': max(seek_times),
                'metrics': self.perf_monitor.get_average_metrics()
            }
            
        return results
    
    def plot_results(self, cpu_results: Dict, memory_results: Dict, disk_results: Dict):
        plt.figure(figsize=(15, 5))
        
        # CPU scheduling results
        plt.subplot(1, 3, 1)
        algorithms = list(cpu_results.keys())
        completion_times = [results['completion_time'] for results in cpu_results.values()]
        plt.bar(algorithms, completion_times)
        plt.title('CPU Scheduling: Completion Time')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        # Memory management results
        plt.subplot(1, 3, 2)
        algorithms = list(memory_results.keys())
        page_faults = [results['page_faults'] for results in memory_results.values()]
        plt.bar(algorithms, page_faults)
        plt.title('Memory Management: Page Faults')
        plt.ylabel('Number of Page Faults')
        plt.xticks(rotation=45)
        
        # Disk scheduling results
        plt.subplot(1, 3, 3)
        algorithms = list(disk_results.keys())
        seek_times = [results['total_seek_time'] for results in disk_results.values()]
        plt.bar(algorithms, seek_times)
        plt.title('Disk Scheduling: Total Seek Time')
        plt.ylabel('Total Seek Time')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        plt.close()

def main():
    benchmark = BenchmarkRunner()
    
    # Run CPU scheduling benchmark
    cpu_results = benchmark.run_cpu_scheduling_benchmark()
    print("\nCPU Scheduling Results:")
    for algorithm, results in cpu_results.items():
        print(f"\n{algorithm.upper()}:")
        print(f"Completion Time: {results['completion_time']:.4f} seconds")
        print("Performance Metrics:", results['metrics'])
    
    # Run memory management benchmark
    memory_results = benchmark.run_memory_management_benchmark()
    print("\nMemory Management Results:")
    for algorithm, results in memory_results.items():
        print(f"\n{algorithm.upper()}:")
        print(f"Completion Time: {results['completion_time']:.4f} seconds")
        print(f"Page Faults: {results['page_faults']}")
        print("Performance Metrics:", results['metrics'])
    
    # Run disk scheduling benchmark
    disk_results = benchmark.run_disk_scheduling_benchmark()
    print("\nDisk Scheduling Results:")
    for algorithm, results in disk_results.items():
        print(f"\n{algorithm.upper()}:")
        print(f"Completion Time: {results['completion_time']:.4f} seconds")
        print(f"Total Seek Time: {results['total_seek_time']}")
        print(f"Average Seek Time: {results['avg_seek_time']:.2f}")
        print(f"Maximum Seek Time: {results['max_seek_time']}")
        print("Performance Metrics:", results['metrics'])
    
    # Plot results
    benchmark.plot_results(cpu_results, memory_results, disk_results)
    print("\nResults have been plotted and saved to 'benchmark_results.png'")

if __name__ == "__main__":
    main()
