"""Traditional disk scheduling algorithms implementation."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class DiskRequest:
    track: int
    arrival_time: int
    size: int  # in KB
    priority: Optional[int] = None

class DiskScheduler:
    def __init__(self, total_tracks: int = 200):
        self.total_tracks = total_tracks
        self.current_track = 0
        self.total_seek_time = 0
        self.direction = 1  # 1 for moving toward higher track numbers, -1 for lower
        
    def seek(self, target_track: int) -> int:
        """Calculate seek time and update current track."""
        seek_time = abs(self.current_track - target_track)
        self.current_track = target_track
        self.total_seek_time += seek_time
        return seek_time

class FCFSDiskScheduler(DiskScheduler):
    """First Come First Serve disk scheduling."""
    def schedule(self, requests: List[DiskRequest]) -> List[int]:
        seek_times = []
        for request in requests:
            seek_time = self.seek(request.track)
            seek_times.append(seek_time)
        return seek_times

class SSTFDiskScheduler(DiskScheduler):
    """Shortest Seek Time First disk scheduling."""
    def schedule(self, requests: List[DiskRequest]) -> List[int]:
        pending_requests = requests.copy()
        seek_times = []
        
        while pending_requests:
            # Find request with minimum seek time
            min_seek_time = float('inf')
            next_request = None
            next_index = 0
            
            for i, request in enumerate(pending_requests):
                seek_time = abs(self.current_track - request.track)
                if seek_time < min_seek_time:
                    min_seek_time = seek_time
                    next_request = request
                    next_index = i
            
            seek_time = self.seek(next_request.track)
            seek_times.append(seek_time)
            pending_requests.pop(next_index)
            
        return seek_times

class SCANDiskScheduler(DiskScheduler):
    """SCAN (Elevator) disk scheduling."""
    def schedule(self, requests: List[DiskRequest]) -> List[int]:
        seek_times = []
        pending_requests = sorted(requests, key=lambda x: x.track)
        
        while pending_requests:
            if self.direction == 1:
                # Moving toward higher track numbers
                next_requests = [r for r in pending_requests if r.track >= self.current_track]
                if not next_requests:
                    self.direction = -1
                    continue
                
                for request in next_requests:
                    seek_time = self.seek(request.track)
                    seek_times.append(seek_time)
                    pending_requests.remove(request)
                    
            else:
                # Moving toward lower track numbers
                next_requests = [r for r in pending_requests if r.track <= self.current_track]
                if not next_requests:
                    self.direction = 1
                    continue
                
                for request in reversed(next_requests):
                    seek_time = self.seek(request.track)
                    seek_times.append(seek_time)
                    pending_requests.remove(request)
                    
        return seek_times

class C_SCANDiskScheduler(DiskScheduler):
    """Circular SCAN disk scheduling."""
    def schedule(self, requests: List[DiskRequest]) -> List[int]:
        seek_times = []
        pending_requests = sorted(requests, key=lambda x: x.track)
        
        while pending_requests:
            # Always move toward higher track numbers
            next_requests = [r for r in pending_requests if r.track >= self.current_track]
            
            if not next_requests:
                # Move to track 0 and continue scanning
                seek_time = self.seek(0)
                seek_times.append(seek_time)
                next_requests = pending_requests
            
            for request in next_requests:
                seek_time = self.seek(request.track)
                seek_times.append(seek_time)
                pending_requests.remove(request)
                
        return seek_times

class LOOKDiskScheduler(DiskScheduler):
    """LOOK disk scheduling."""
    def schedule(self, requests: List[DiskRequest]) -> List[int]:
        seek_times = []
        pending_requests = sorted(requests, key=lambda x: x.track)
        
        while pending_requests:
            if self.direction == 1:
                next_requests = [r for r in pending_requests if r.track >= self.current_track]
                if not next_requests:
                    self.direction = -1
                    continue
                
                for request in next_requests:
                    seek_time = self.seek(request.track)
                    seek_times.append(seek_time)
                    pending_requests.remove(request)
            else:
                next_requests = [r for r in pending_requests if r.track <= self.current_track]
                if not next_requests:
                    self.direction = 1
                    continue
                
                for request in reversed(next_requests):
                    seek_time = self.seek(request.track)
                    seek_times.append(seek_time)
                    pending_requests.remove(request)
                    
        return seek_times
