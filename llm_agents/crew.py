"""Multi-agent crew for OS optimization."""

import os
from typing import List, Dict, Any
from openai import OpenAI
from .cpu_agent import CPUSchedulerAgent
from .memory_agent import MemoryManagerAgent
from .disk_agent import DiskSchedulerAgent
from benchmarks.workload_generator import WorkloadTask
from traditional.disk_scheduling import DiskRequest
import json

class OSOptimizationCrew:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.token = os.environ.get("GITHUB_TOKEN")
        self.endpoint = "https://models.inference.ai.azure.com"
        
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable not set")
            
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )
        
        self.cpu_agent = CPUSchedulerAgent(self.client, self.model)
        self.memory_agent = MemoryManagerAgent(num_frames=10, client=self.client, model=self.model)
        self.disk_agent = DiskSchedulerAgent(self.client, self.model)
        
        # Coordinator agent for high-level decisions
        self.coordinator_prompt = """You are the coordinator of an OS optimization system.
        Your task is to analyze the decisions of specialized agents and ensure they work together effectively.
        Consider:
        1. Resource dependencies
        2. System bottlenecks
        3. Overall performance goals
        4. Trade-offs between different resources
        
        Output your analysis as a JSON with:
        {
            "approved": true/false,
            "adjustments": {
                "cpu": {...},
                "memory": {...},
                "disk": {...}
            },
            "reasoning": "explanation"
        }
        """
        
    def coordinate_decisions(self, 
                           cpu_decision: Dict[str, Any],
                           memory_decision: Dict[str, Any],
                           disk_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate decisions from different agents."""
        prompt = f"""Current agent decisions:
        CPU: {cpu_decision.get('strategy', 'Unknown')}
        Memory: {memory_decision.get('strategy', 'Unknown')}
        Disk: {disk_decision.get('strategy', 'Unknown')}
        
        Should we approve these decisions? Respond with a single character:
        'Y' - Yes, approve all decisions
        'C' - Modify CPU decision
        'M' - Modify memory decision
        'D' - Modify disk decision
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.coordinator_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                stream=False
            )
            
            decision = response.choices[0].message.content.strip()
            
            if decision == 'Y':
                return {
                    "approved": True,
                    "adjustments": {},
                    "reasoning": "All decisions approved"
                }
            elif decision == 'C':
                return {
                    "approved": False,
                    "adjustments": {
                        "cpu": {"strategy": "RR"}  # Default to Round Robin
                    },
                    "reasoning": "CPU schedule needs adjustment"
                }
            elif decision == 'M':
                return {
                    "approved": False,
                    "adjustments": {
                        "memory": {"strategy": "LRU"}  # Default to LRU
                    },
                    "reasoning": "Memory management needs adjustment"
                }
            elif decision == 'D':
                return {
                    "approved": False,
                    "adjustments": {
                        "disk": {"strategy": "SSTF"}  # Default to SSTF
                    },
                    "reasoning": "Disk schedule needs adjustment"
                }
            else:
                return {
                    "approved": True,
                    "adjustments": {},
                    "reasoning": "Default approval"
                }
            
        except Exception as e:
            print(f"Error in coordination: {e}")
            return {
                "approved": True,
                "adjustments": {},
                "reasoning": "Fallback: proceeding with original decisions"
            }
            
    def optimize_system(self,
                       cpu_workload: List[WorkloadTask],
                       memory_pages: List[int],
                       disk_requests: List[DiskRequest],
                       current_time: int) -> Dict[str, Any]:
        """Optimize all system resources together."""
        # Get individual agent decisions
        cpu_decision = self.cpu_agent.optimize(cpu_workload, current_time)
        memory_decision = self.memory_agent.optimize(memory_pages[0] if memory_pages else 0)
        disk_decision = self.disk_agent.optimize(disk_requests)
        
        # Coordinate decisions
        coordination = self.coordinate_decisions(cpu_decision, memory_decision, disk_decision)
        
        # Apply any adjustments from coordination
        if coordination["approved"]:
            adjustments = coordination["adjustments"]
            if "cpu" in adjustments:
                cpu_decision.update(adjustments["cpu"])
            if "memory" in adjustments:
                memory_decision.update(adjustments["memory"])
            if "disk" in adjustments:
                disk_decision.update(adjustments["disk"])
                
        return {
            "cpu": cpu_decision,
            "memory": memory_decision,
            "disk": disk_decision,
            "coordination": coordination["reasoning"]
        }
