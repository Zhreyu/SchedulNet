"""LLM agent for memory management."""

from typing import List, Dict, Any
import numpy as np
import json
from openai import OpenAI
import random

class MemoryManagerAgent:
    def __init__(self, client: OpenAI, model: str = "gpt-4o", num_frames: int = 10):
        self.client = client
        self.model = model
        self.num_frames = num_frames
        self.frames = []
        self.page_faults = 0
        self.access_history = []
        
        self.prompt = """You are a memory management agent.
        Your task is to decide which page to replace when a page fault occurs.
        Respond with ONLY a single character:
        'L' - Replace least recently used page
        'F' - Replace first loaded page (FIFO)
        'R' - Replace random page
        'O' - Replace optimal page based on future accesses
        """
        
    def optimize(self, page: int) -> Dict[str, Any]:
        """Optimize page replacement for the given page access."""
        self.access_history.append(page)
        
        if page in self.frames:
            return {"hit": True, "page": page, "frame_index": self.frames.index(page)}
            
        self.page_faults += 1
        
        if len(self.frames) < self.num_frames:
            self.frames.append(page)
            return {
                "hit": False,
                "page": page,
                "frame_index": len(self.frames) - 1,
                "reason": "Frame available"
            }
        
        prompt = f"""Current page access: {page}
        Frames: {self.frames}
        Recent access history: {self.access_history[-10:]}
        Which page replacement strategy should be used? Respond with a single character.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                stream=False
            )
            
            decision = response.choices[0].message.content.strip()
            
            # Convert single character decision to replacement strategy
            if decision == 'L':
                # LRU: replace page that hasn't been accessed for longest time
                lru_page = min(self.frames, key=lambda x: len(self.access_history) - 1 - max(i for i, p in enumerate(self.access_history) if p == x))
                frame_index = self.frames.index(lru_page)
                strategy = "LRU"
            elif decision == 'F':
                # FIFO: replace first page loaded
                frame_index = 0
                strategy = "FIFO"
            elif decision == 'R':
                # Random: replace random page
                frame_index = random.randint(0, len(self.frames) - 1)
                strategy = "Random"
            else:  # 'O' or fallback
                # Optimal: try to look at future accesses (simplified)
                frame_index = 0
                strategy = "Optimal"
            
            old_page = self.frames[frame_index]
            self.frames[frame_index] = page
            
            return {
                "hit": False,
                "page": page,
                "frame_index": frame_index,
                "evicted_page": old_page,
                "strategy": strategy,
                "metrics": {
                    "page_faults": self.page_faults,
                    "fault_rate": self.page_faults / len(self.access_history)
                }
            }
            
        except Exception as e:
            print(f"Error in memory management: {e}")
            # Fallback to FIFO
            old_page = self.frames[0]
            self.frames[0] = page
            return {
                "hit": False,
                "page": page,
                "frame_index": 0,
                "evicted_page": old_page,
                "strategy": "FIFO (fallback)",
                "metrics": {
                    "page_faults": self.page_faults,
                    "fault_rate": self.page_faults / len(self.access_history)
                }
            }
