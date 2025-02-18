"""Traditional memory management algorithms implementation."""

class PageReplacementAlgorithm:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frames = []
        self.page_faults = 0
        
    def access_page(self, page):
        raise NotImplementedError

class FIFOPageReplacement(PageReplacementAlgorithm):
    def access_page(self, page):
        if page not in self.frames:
            self.page_faults += 1
            if len(self.frames) >= self.num_frames:
                self.frames.pop(0)  # Remove oldest page
            self.frames.append(page)
        return self.page_faults

class LRUPageReplacement(PageReplacementAlgorithm):
    def access_page(self, page):
        if page in self.frames:
            # Move accessed page to the end (most recently used)
            self.frames.remove(page)
            self.frames.append(page)
        else:
            self.page_faults += 1
            if len(self.frames) >= self.num_frames:
                self.frames.pop(0)  # Remove least recently used page
            self.frames.append(page)
        return self.page_faults
