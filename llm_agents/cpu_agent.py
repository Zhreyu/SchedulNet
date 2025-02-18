"""LLM agent for CPU scheduling."""

import json
from typing import List, Dict, Any
from openai import OpenAI
from traditional.cpu_scheduling import Process

class CPUSchedulerAgent:
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        self.system_prompt = """You are a CPU scheduling agent.
        Your task is to decide which process to schedule next.
        Respond with ONLY a single character:
        'S' - Schedule shortest job
        'L' - Schedule longest job
        'P' - Schedule highest priority
        'R' - Schedule round-robin
        """
        
    def optimize(self, workload: List[Process], current_time: int) -> Dict[str, Any]:
        """Optimize CPU scheduling for given workload."""
        # Prepare workload data for LLM
        workload_info = []
        for task in workload:
            if not task.completed:
                workload_info.append({
                    "process_id": task.pid,
                    "arrival_time": task.arrival_time,
                    "burst_time": task.burst_time,
                    "priority": task.priority
                })
                
        prompt = f"""Current time: {current_time}
        Active processes: {len(workload_info)}
        Process details: {json.dumps(workload_info)}
        Which scheduling strategy should be used? Respond with a single character.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                stream=False
            )
            
            decision = response.choices[0].message.content.strip()
            
            # Convert single character decision to scheduling action
            if decision == 'S':
                next_process = min(workload, key=lambda x: x.burst_time)
                strategy = "Shortest Job First"
            elif decision == 'L':
                next_process = max(workload, key=lambda x: x.burst_time)
                strategy = "Longest Job First"
            elif decision == 'P':
                next_process = max(workload, key=lambda x: x.priority)
                strategy = "Highest Priority"
            else:  # 'R' or fallback
                next_process = min(workload, key=lambda x: x.arrival_time)
                strategy = "Round Robin"
            
            # Calculate simple metrics
            completed_processes = [p for p in workload if hasattr(p, 'completed') and p.completed]
            waiting_time = current_time - next_process.arrival_time if hasattr(next_process, 'arrival_time') else 0
            
            return {
                "next_process": next_process.pid,
                "time_slice": min(next_process.burst_time, 10),  # Cap at 10 time units
                "metrics": {
                    "avg_completion_time": waiting_time,
                    "max_completion_time": max((current_time - p.arrival_time for p in completed_processes), default=0),
                    "throughput": len(completed_processes) / (current_time + 1) if current_time > 0 else 0
                },
                "strategy": strategy
            }
            
        except Exception as e:
            print(f"Error in CPU scheduling: {e}")
            # Fallback to FCFS
            next_process = min(workload, key=lambda x: x.arrival_time)
            return {
                "next_process": next_process.pid,
                "time_slice": next_process.burst_time,
                "metrics": {
                    "avg_completion_time": float('inf'),
                    "max_completion_time": float('inf'),
                    "throughput": 0
                },
                "strategy": "FCFS (fallback)"
            }
