
from abc import abstractmethod
from typing import Dict, List, Tuple, Type, Optional, Union
from collections import deque
from random import sample

import numpy as np

class ReplayExample():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class HeuristicReplayExample(ReplayExample):

    # Replay example that has an associated heuristic, to be used for preferential replay mechanisms

    def __init__(self, x, y, h_value):
        super().__init__(x, y)
        self.h_value = h_value

class ReplayBufferInterface():
    
    @abstractmethod
    def add_examples(self, examples: List[ReplayExample]):
        raise NotImplementedError

    @abstractmethod
    def get_examples(self, num_examples: int, random: bool = True):
        raise NotImplementedError

class NonFunctionalReplayBuffer(ReplayBufferInterface):

    # "Implementation" of replay buffer that does not do anything

    def add_examples(self, examples: List[ReplayExample]):
        pass

    def get_examples(self, num_examples: int, random: bool = True):
        return []

class FixedLengthReplayBuffer(ReplayBufferInterface):
    
    def __init__(self, buffer_size: int = 5_000):
        self.buffer = list()  # initialise empty buffer
        self.buffer_size = 5_000

    def add_examples(self, examples: List[ReplayExample]):
        self.buffer.extend(examples)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]  # remove front (oldest) elements

    def get_examples(self, num_examples: int, random: bool = True):

        if num_examples > len(self.buffer):
            return []

        if random:
            return sample(list(self.buffer), num_examples)
        else:
            return list(self.buffer)[-num_examples:]

class HeuristicSortedReplayBuffer(FixedLengthReplayBuffer):

    # Replay buffer where the replay examples are sorted by some heuristic
    #  - Replay examples that have higher / lower heuristic (depending on if reverse sort or not)
    #    will be in memory for more time, thus they will influence model dynamics more.

    def __init__(self, reverse_sort: bool = False, buffer_size: int = 5_000):
        super().__init__(buffer_size)
        self.reverse_sort = reverse_sort

    def add_examples(self, examples: List[HeuristicReplayExample]):
        # add examples to buffer and sort based on hinge loss
        # remove examples at bottom of list if list exceeds maximum size
        self.buffer.extend(examples)
        self.buffer.sort(key = lambda ex: ex.h_value, reverse = self.reverse_sort)
        
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]  # remove examples with smallest / largest hinge loss (depending on if reversed)