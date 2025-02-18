from dataclasses import dataclass
from typing import Optional

@dataclass
class Process:
    pid: int
    arrival_time: int
    burst_time: int
    priority: int
    completed: bool = False
    completion_time: Optional[int] = None
    
    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

@dataclass
class Page:
    page_id: int
    size: int
    is_fault: bool
    last_access: int
    loaded_time: int
    referenced: bool = False
    
    def __lt__(self, other):
        return self.last_access < other.last_access

@dataclass
class DiskRequest:
    request_id: int
    track: int
    arrival_time: int
    size: int
    completed: bool = False
    
    def __lt__(self, other):
        return self.track < other.track
