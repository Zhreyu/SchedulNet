import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from drl_agents import CPUSchedulerAgent, MemoryManagerAgent, DiskSchedulerAgent, CrewCoordinator
from drl_agents.models import Process, Page, DiskRequest

class RLBenchmark:
    def __init__(self):
        self.cpu_agent = CPUSchedulerAgent()
        self.memory_agent = MemoryManagerAgent()
        self.disk_agent = DiskSchedulerAgent()
        self.coordinator = CrewCoordinator()
        
    def generate_workload(self, num_tasks: int) -> Dict[str, List[Any]]:
        # Generate synthetic workload
        cpu_tasks = []
        memory_pages = {}
        disk_requests = []
        
        for i in range(num_tasks):
            # CPU task
            cpu_tasks.append(Process(
                pid=i,
                arrival_time=np.random.randint(0, 100),
                burst_time=np.random.randint(1, 20),
                priority=np.random.randint(1, 10)
            ))
            
            # Memory pages
            memory_pages[i] = Page(
                page_id=i,
                size=np.random.randint(1, 8) * 1024,  # 1-8 KB
                is_fault=np.random.random() < 0.2,
                last_access=np.random.randint(0, 100),
                loaded_time=np.random.randint(0, 100)
            )
            
            # Disk requests
            disk_requests.append(DiskRequest(
                request_id=i,
                track=np.random.randint(0, 1000),
                arrival_time=np.random.randint(0, 100),
                size=np.random.randint(1, 16) * 512  # 512B - 8KB
            ))
            
        return {
            'cpu': cpu_tasks,
            'memory': list(memory_pages.values()),
            'disk': disk_requests
        }
        
    def run_benchmarks(self, num_tasks: int = 100, episodes: int = 10) -> Dict[str, Any]:
        results = {
            'cpu': {'completion_times': [], 'throughput': []},
            'memory': {'fault_rates': []},
            'disk': {'seek_times': []}
        }
        
        for episode in range(episodes):
            workload = self.generate_workload(num_tasks)
            current_time = 0
            
            # Run agents
            cpu_decision = self.cpu_agent.optimize(workload['cpu'], current_time)
            memory_decision = self.memory_agent.optimize(workload['memory'], [], current_time)
            disk_decision = self.disk_agent.optimize(workload['disk'], current_time)
            
            # Coordinator reviews decisions
            coordination = self.coordinator.coordinate_decisions(
                cpu_decision, memory_decision, disk_decision
            )
            
            # Record results
            results['cpu']['completion_times'].append(cpu_decision['metrics']['avg_completion_time'])
            results['cpu']['throughput'].append(cpu_decision['metrics']['throughput'])
            results['memory']['fault_rates'].append(memory_decision['metrics']['fault_rate'])
            results['disk']['seek_times'].append(disk_decision['metrics']['avg_seek_time'])
            
        # Calculate averages
        avg_results = {
            'cpu': {
                'avg_completion_time': np.mean(results['cpu']['completion_times']),
                'avg_throughput': np.mean(results['cpu']['throughput'])
            },
            'memory': {
                'avg_fault_rate': np.mean(results['memory']['fault_rates'])
            },
            'disk': {
                'avg_seek_time': np.mean(results['disk']['seek_times'])
            }
        }
        
        self.plot_results(results)
        return avg_results
        
    def plot_results(self, results: Dict[str, Any]):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # CPU Performance
        ax1.plot(results['cpu']['completion_times'], label='Completion Time')
        ax1.plot(results['cpu']['throughput'], label='Throughput')
        ax1.set_title('CPU Performance')
        ax1.set_xlabel('Episode')
        ax1.legend()
        
        # Memory Performance
        ax2.plot(results['memory']['fault_rates'], label='Page Fault Rate')
        ax2.set_title('Memory Performance')
        ax2.set_xlabel('Episode')
        ax2.legend()
        
        # Disk Performance
        ax3.plot(results['disk']['seek_times'], label='Avg Seek Time')
        ax3.set_title('Disk Performance')
        ax3.set_xlabel('Episode')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('benchmarks/rl_results.png')
        plt.close()
