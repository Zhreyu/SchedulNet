"""Benchmark comparing Traditional algorithms with LLM agents."""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from benchmarks.workload_generator import WorkloadGenerator, WorkloadTask
from traditional.cpu_scheduling import RoundRobinScheduler, SJFScheduler, Process
from traditional.memory_management import LRUPageReplacement, FIFOPageReplacement
from traditional.disk_scheduling import FCFSDiskScheduler, SSTFDiskScheduler, DiskRequest
from llm_agents import OSOptimizationCrew

class ComprehensiveBenchmark:
    def __init__(self):
        self.workload_gen = WorkloadGenerator()
        
        # Traditional schedulers
        self.trad_cpu = {
            'RR': RoundRobinScheduler(time_quantum=2),
            'SJF': SJFScheduler()
        }
        self.trad_memory = {
            'LRU': LRUPageReplacement(num_frames=10),
            'FIFO': FIFOPageReplacement(num_frames=10)
        }
        self.trad_disk = {
            'FCFS': FCFSDiskScheduler(),
            'SSTF': SSTFDiskScheduler()
        }
        
        # LLM agents
        self.llm_crew = OSOptimizationCrew()
        
    def run_benchmarks(self, num_tasks: int = 100) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks comparing traditional algorithms with LLM agents."""
        # Generate workloads
        cpu_workload = self.workload_gen.generate_cpu_workload(
            num_tasks=num_tasks,
            arrival_time_range=(0, 100),
            burst_time_range=(1, 20),
            memory_range=(64, 1024)
        )
        
        memory_workload = self.workload_gen.generate_memory_workload(
            num_pages=num_tasks * 10,
            page_range=(0, 100),
            page_size_range=(4, 64)
        )
        
        disk_workload = self.workload_gen.generate_disk_workload(
            num_requests=num_tasks,
            track_range=(0, 200),
            arrival_time_range=(0, 100),
            request_size_range=(4, 64)
        )
        
        # Run traditional benchmarks
        trad_results = self.run_traditional_benchmark(
            workload=cpu_workload,
            memory_pages=[page for page, _ in memory_workload],
            disk_requests=disk_workload
        )
        
        # Run LLM benchmarks
        llm_results = self.run_llm_benchmark(
            workload=cpu_workload,
            memory_pages=[page for page, _ in memory_workload],
            disk_requests=disk_workload
        )
        
        # Calculate improvement percentages
        improvements = self.calculate_improvements(trad_results, llm_results)
        
        return {
            'traditional': trad_results,
            'llm': llm_results,
            'improvements': improvements
        }
        
    def calculate_improvements(self, trad_results: Dict[str, Any], llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement percentages of LLM over traditional algorithms."""
        improvements = {}
        
        # CPU improvements
        trad_cpu_avg = min(
            trad_results['cpu']['RR']['avg_completion_time'],
            trad_results['cpu']['SJF']['avg_completion_time']
        )
        llm_cpu_avg = llm_results['cpu']['LLM']['avg_completion_time']
        improvements['cpu'] = {
            'avg_completion_time': ((trad_cpu_avg - llm_cpu_avg) / trad_cpu_avg) * 100,
            'throughput': ((llm_results['cpu']['LLM']['throughput'] / 
                           max(trad_results['cpu']['RR']['throughput'],
                               trad_results['cpu']['SJF']['throughput'])) - 1) * 100
        }
        
        # Memory improvements
        trad_mem_rate = min(
            trad_results['memory']['LRU']['fault_rate'],
            trad_results['memory']['FIFO']['fault_rate']
        )
        llm_mem_rate = llm_results['memory']['LLM']['fault_rate']
        improvements['memory'] = {
            'fault_rate': ((trad_mem_rate - llm_mem_rate) / trad_mem_rate) * 100
        }
        
        # Disk improvements
        trad_disk_avg = min(
            trad_results['disk']['FCFS']['avg_seek_time'],
            trad_results['disk']['SSTF']['avg_seek_time']
        )
        llm_disk_avg = llm_results['disk']['LLM']['avg_seek_time']
        improvements['disk'] = {
            'avg_seek_time': ((trad_disk_avg - llm_disk_avg) / trad_disk_avg) * 100,
            'total_seek_time': ((min(trad_results['disk']['FCFS']['total_seek_time'],
                                   trad_results['disk']['SSTF']['total_seek_time']) -
                                llm_results['disk']['LLM']['total_seek_time']) /
                               min(trad_results['disk']['FCFS']['total_seek_time'],
                                   trad_results['disk']['SSTF']['total_seek_time'])) * 100
        }
        
        return improvements
        
    def run_traditional_benchmark(self, workload, memory_pages, disk_requests) -> Dict[str, Any]:
        """Run traditional scheduling algorithms."""
        results = {'cpu': {}, 'memory': {}, 'disk': {}}
        
        # Convert workload to Process objects
        processes = [
            Process(
                pid=task.task_id,
                arrival_time=task.arrival_time,
                burst_time=task.burst_time,
                priority=task.priority
            )
            for task in workload
        ]
        
        # CPU scheduling
        for name, scheduler in self.trad_cpu.items():
            start_time = time.time()
            scheduler.schedule(processes)
            end_time = time.time()
            
            completion_times = [p.burst_time + p.arrival_time for p in scheduler.completed_processes]
            
            results['cpu'][name] = {
                'avg_completion_time': np.mean(completion_times),
                'max_completion_time': np.max(completion_times),
                'throughput': len(workload) / (end_time - start_time),
                'execution_time': end_time - start_time
            }
            
        # Memory scheduling
        for name, scheduler in self.trad_memory.items():
            start_time = time.time()
            for page in memory_pages:
                scheduler.access_page(page)
            end_time = time.time()
            
            results['memory'][name] = {
                'page_faults': scheduler.page_faults,
                'fault_rate': scheduler.page_faults / len(memory_pages),
                'execution_time': end_time - start_time
            }
            
        # Disk scheduling
        for name, scheduler in self.trad_disk.items():
            start_time = time.time()
            seek_times = scheduler.schedule(disk_requests)
            end_time = time.time()
            
            results['disk'][name] = {
                'avg_seek_time': np.mean(seek_times),
                'total_seek_time': np.sum(seek_times),
                'execution_time': end_time - start_time
            }
            
        return results
    
    def run_llm_benchmark(self, workload, memory_pages, disk_requests) -> Dict[str, Any]:
        """Run LLM-based scheduling algorithms."""
        results = {'cpu': {}, 'memory': {}, 'disk': {}}
        
        # Convert workload to Process objects
        processes = [
            Process(
                pid=task.task_id,
                arrival_time=task.arrival_time,
                burst_time=task.burst_time,
                priority=task.priority
            )
            for task in workload
        ]
        
        # Run LLM agents
        start_time = time.time()
        llm_results = self.llm_crew.optimize_system(
            cpu_workload=processes,
            memory_pages=memory_pages,
            disk_requests=disk_requests,
            current_time=int(time.time())
        )
        end_time = time.time()
        
        # Extract metrics from LLM results
        if 'cpu' in llm_results:
            results['cpu']['LLM'] = {
                'avg_completion_time': llm_results['cpu'].get('avg_completion_time', 0),
                'max_completion_time': llm_results['cpu'].get('max_completion_time', 0),
                'throughput': llm_results['cpu'].get('throughput', 0),
                'execution_time': end_time - start_time
            }
            
        if 'memory' in llm_results:
            results['memory']['LLM'] = {
                'page_faults': llm_results['memory'].get('page_faults', 0),
                'fault_rate': llm_results['memory'].get('fault_rate', 0),
                'execution_time': end_time - start_time
            }
            
        if 'disk' in llm_results:
            results['disk']['LLM'] = {
                'avg_seek_time': llm_results['disk'].get('avg_seek_time', 0),
                'total_seek_time': llm_results['disk'].get('total_seek_time', 0),
                'execution_time': end_time - start_time
            }
        
        return results
    
    def plot_results(self, results: Dict[str, Dict[str, Any]]):
        """Plot benchmark results for visualization."""
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot CPU scheduling results
        ax = axes[0, 0]
        cpu_data = {
            'Traditional (Best)': min(
                results['traditional']['cpu']['RR']['avg_completion_time'],
                results['traditional']['cpu']['SJF']['avg_completion_time']
            ),
            'LLM': results['llm']['cpu']['LLM']['avg_completion_time']
        }
        ax.bar(cpu_data.keys(), cpu_data.values())
        ax.set_title('CPU Scheduling: Average Completion Time')
        ax.set_ylabel('Time')
        ax.tick_params(axis='x', rotation=45)
        
        # Add improvement percentage
        improvement = results['improvements']['cpu']['avg_completion_time']
        ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
                transform=ax.transAxes, ha='center')
        
        # Plot Memory management results
        ax = axes[0, 1]
        memory_data = {
            'Traditional (Best)': min(
                results['traditional']['memory']['LRU']['fault_rate'],
                results['traditional']['memory']['FIFO']['fault_rate']
            ),
            'LLM': results['llm']['memory']['LLM']['fault_rate']
        }
        ax.bar(memory_data.keys(), memory_data.values())
        ax.set_title('Memory Management: Page Fault Rate')
        ax.set_ylabel('Fault Rate')
        ax.tick_params(axis='x', rotation=45)
        
        # Add improvement percentage
        improvement = results['improvements']['memory']['fault_rate']
        ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
                transform=ax.transAxes, ha='center')
        
        # Plot Disk scheduling results
        ax = axes[1, 0]
        disk_data = {
            'Traditional (Best)': min(
                results['traditional']['disk']['FCFS']['avg_seek_time'],
                results['traditional']['disk']['SSTF']['avg_seek_time']
            ),
            'LLM': results['llm']['disk']['LLM']['avg_seek_time']
        }
        ax.bar(disk_data.keys(), disk_data.values())
        ax.set_title('Disk Scheduling: Average Seek Time')
        ax.set_ylabel('Time')
        ax.tick_params(axis='x', rotation=45)
        
        # Add improvement percentage
        improvement = results['improvements']['disk']['avg_seek_time']
        ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
                transform=ax.transAxes, ha='center')
        
        # Plot Overall Improvements
        ax = axes[1, 1]
        improvements = {
            'CPU Completion Time': results['improvements']['cpu']['avg_completion_time'],
            'Memory Fault Rate': results['improvements']['memory']['fault_rate'],
            'Disk Seek Time': results['improvements']['disk']['avg_seek_time']
        }
        colors = ['g' if v > 0 else 'r' for v in improvements.values()]
        ax.bar(improvements.keys(), improvements.values(), color=colors)
        ax.set_title('Overall Improvements')
        ax.set_ylabel('Improvement (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        plt.close()
