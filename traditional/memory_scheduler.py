"""Traditional memory management algorithms."""

from typing import List
from collections import OrderedDict

class LRUScheduler:
    """Least Recently Used page replacement."""
    
    def __init__(self, num_frames: int = 10):
        self.num_frames = num_frames
        self.frames = OrderedDict()
        
    def schedule(self, pages: List[int]) -> int:
        """Schedule page accesses using LRU."""
        page_faults = 0
        
        for page in pages:
            if page not in self.frames:
                page_faults += 1
                
                if len(self.frames) >= self.num_frames:
                    # Remove least recently used
                    self.frames.popitem(last=False)
                    
            else:
                # Remove and re-add to maintain order
                self.frames.pop(page)
                
            self.frames[page] = True
            
        return page_faults

class FIFOMemoryScheduler:
    """First In First Out page replacement."""
    
    def __init__(self, num_frames: int = 10):
        self.num_frames = num_frames
        self.frames = []
        
    def schedule(self, pages: List[int]) -> int:
        """Schedule page accesses using FIFO."""
        page_faults = 0
        
        for page in pages:
            if page not in self.frames:
                page_faults += 1
                
                if len(self.frames) >= self.num_frames:
                    # Remove oldest page
                    self.frames.pop(0)
                    
                self.frames.append(page)
                
        return page_faults
