import numpy as np


class RingBuffer(object):

    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size)
        self.index = 0
        self.count = 0

    def add(self, val):
        self.buffer[self.index] = val
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def empty(self):
        return self.count == 0

    def clear(self):
        self.buffer *= 0
        self.index = 0
        self.count = 0

    def full(self):
        return self.count == self.size

    def mean(self):
        return self.buffer[:self.count].mean()

    def std(self):
        return self.buffer[:self.count].std()

    def median(self):
        return np.median(self.buffer[:self.count])
