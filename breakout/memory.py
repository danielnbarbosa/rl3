import random


class ReplayMemory:
    '''FIFO buffer for storing experience tuples.'''

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.idx = 0

    def add(self, item):
        '''Add an item to the buffer.'''
        if len(self.memory) < self.size:
            self.memory.append(item)
        else:
            self.memory[self.idx] = item
        self.idx = (self.idx + 1) % self.size  # circulur buffer

    def sample(self, n):
        '''Randomly sample n elements from the buffer'''
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)
