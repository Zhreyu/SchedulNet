"""AutoGen-based OS optimization crew."""

import autogen
from typing import List, Dict, Any
from .agent_config import get_config, AGENT_CONFIGS
from benchmarks.workload_generator import WorkloadTask
from traditional.disk_scheduling import DiskRequest

class OSOptimizationCrew:
    def __init__(self, config_list: List[Dict[str, Any]] = None):
        # Initialize configurations
        self.config = config_list or get_config()
        
        # Create agents
        self.cpu_scheduler = autogen.AssistantAgent(
            name=AGENT_CONFIGS["cpu_scheduler"]["name"],
            system_message=AGENT_CONFIGS["cpu_scheduler"]["system_message"],
            llm_config=self.config
        )
        
        self.memory_manager = autogen.AssistantAgent(
            name=AGENT_CONFIGS["memory_manager"]["name"],
            system_message=AGENT_CONFIGS["memory_manager"]["system_message"],
            llm_config=self.config
        )
        
        self.disk_scheduler = autogen.AssistantAgent(
            name=AGENT_CONFIGS["disk_scheduler"]["name"],
            system_message=AGENT_CONFIGS["disk_scheduler"]["system_message"],
            llm_config=self.config
        )
        
        self.performance_analyst = autogen.AssistantAgent(
            name=AGENT_CONFIGS["performance_analyst"]["name"],
            system_message=AGENT_CONFIGS["performance_analyst"]["system_message"],
            llm_config=self.config
        )
        
        # Create user proxy for coordination
        self.coordinator = autogen.UserProxyAgent(
            name="OS Coordinator",
            system_message="Coordinate between different OS components to optimize overall system performance.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10
        )
        
        # Create group chat
        self.group_chat = autogen.GroupChat(
            agents=[
                self.coordinator,
                self.cpu_scheduler,
                self.memory_manager,
                self.disk_scheduler,
                self.performance_analyst
            ],
            messages=[],
            max_round=5
        )
        
        # Create group chat manager
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.config
        )
        
    def optimize_system(self,
                       cpu_workload: List[WorkloadTask],
                       memory_pages: List[int],
                       disk_requests: List[DiskRequest],
                       current_time: int) -> Dict[str, Any]:
        """Optimize system resources using multi-agent collaboration."""
        
        # Prepare system state message
        system_state = {
            "current_time": current_time,
            "cpu_workload": [
                {
                    "id": task.task_id,
                    "arrival": task.arrival_time,
                    "burst": task.burst_time,
                    "priority": task.priority,
                    "memory": task.memory_size
                }
                for task in cpu_workload
            ],
            "memory_state": {
                "pages": memory_pages,
                "access_count": len(memory_pages)
            },
            "disk_state": [
                {
                    "track": req.track,
                    "arrival": req.arrival_time,
                    "size": req.size,
                    "priority": req.priority
                }
                for req in disk_requests
            ]
        }
        
        # Initialize chat with system state
        self.coordinator.initiate_chat(
            self.manager,
            message=f"""Current system state:
            {system_state}
            
            Please analyze and optimize the system performance. Each agent should:
            1. Analyze their specific component
            2. Propose optimizations
            3. Consider impact on other components
            4. Collaborate on overall system performance
            
            Required: Final output must be a JSON with specific scheduling decisions."""
        )
        
        # Extract decisions from chat
        last_message = self.group_chat.messages[-1]["content"]
        try:
            import json
            decisions = json.loads(last_message)
            return decisions
        except Exception as e:
            print(f"Error parsing decisions: {e}")
            return {
                "error": "Failed to parse optimization decisions",
                "fallback": "Using default scheduling algorithms"
            }
