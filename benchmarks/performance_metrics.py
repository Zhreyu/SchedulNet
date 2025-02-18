"""Performance measurement and comparison utilities."""

import time
import psutil
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    io_wait: float
    context_switches: int
    timestamp: float

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        
    def collect_metrics(self) -> SystemMetrics:
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        io_counters = psutil.disk_io_counters()
        context_switches = psutil.cpu_stats().ctx_switches
        
        metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            io_wait=psutil.cpu_times_percent().iowait,
            context_switches=context_switches,
            timestamp=time.time()
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_average_metrics(self, window_size: int = None) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
            
        if window_size:
            metrics = self.metrics_history[-window_size:]
        else:
            metrics = self.metrics_history
            
        return {
            'avg_cpu_usage': np.mean([m.cpu_usage for m in metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in metrics]),
            'avg_io_wait': np.mean([m.io_wait for m in metrics]),
            'context_switches_per_sec': np.mean([m.context_switches for m in metrics[1:]]) - \
                                      np.mean([m.context_switches for m in metrics[:-1]])
        }
    
    def clear_history(self):
        self.metrics_history.clear()
