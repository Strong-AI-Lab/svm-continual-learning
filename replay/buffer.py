
#import torch

from abc import abstractmethod
from typing import Collection, Dict, List, Tuple, Type, Optional, Union
from random import sample

from .heuristic import Heuristic
from .context import SharedStepContext

import numpy as np

class ReplayExample():
    def __init__(self, x, y, heuristics: List[Heuristic]):
        self.x = x
        self.y = y

        self.heuristics_val = 0  # last calculated value of combined heuristic
        self.heuristics = heuristics   # list of the heuristic objects for this example\

    def update_heuristics(self, context: SharedStepContext, **kwargs):
        for heuristic in self.heuristics:
            heuristic.calculate(context)  # TODO: pass on kwargs

        # for now just add heuristics together
        # TODO: enable other means of combination
        self.heuristics_val = np.mean([heuristic.val for heuristic in self.heuristics])

class AbstractReplayBuffer():

    def __init__(self, heuristic_classes: List[Type[Heuristic]]):
        self.buffer: Collection
        # heuristics is a list of Heuristic derivative classes which are used to instantiate
        # heuristic objects for new replay examples
        self.heuristic_classes = heuristic_classes

    @abstractmethod
    def _add_examples(self, example: ReplayExample):
        # How an individual example is added will depend upon the buffer container used (e.g. heap, list, etc.)
        #  - Also, some methods may want to add a 'cut-off' heuristic score or something similar to filter replay examples
        raise NotImplementedError
    
    def add_examples(self, context: SharedStepContext):
        
        new_examples = []

        for i, x_example in enumerate(context.x):

            context.ex_i = i
            heuristics = [heuristic_class() for heuristic_class in self.heuristic_classes]

            example = ReplayExample(x_example, context.y_targets[i], heuristics)
            example.update_heuristics(context)
            new_examples.append(example)
        
        self._add_examples(new_examples)

    @abstractmethod
    def get_examples(self, num_examples: int, random: bool = True):
        raise NotImplementedError

class NonFunctionalReplayBuffer(AbstractReplayBuffer):

    # "Implementation" of replay buffer that does not do anything

    def __init__(self):
        super().__init__([])

    def add_examples(self, examples: List[ReplayExample]):
        pass

    def get_examples(self, num_examples: int, random: bool = True):
        return []

class HeuristicSortedReplayBuffer(AbstractReplayBuffer):

    # Replay buffer where the replay examples are sorted by some heuristic
    #  - Replay examples that have higher / lower heuristic (depending on if reverse sort or not)
    #    will be in memory for more time, thus they will influence model dynamics more.

    def __init__(self, heuristics: List[Type[Heuristic]], max_buffer_size: int = 5_000, reverse_sort: bool = False):
        super().__init__(heuristics)

        self.buffer = list()
        self.max_buffer_size = max_buffer_size
        self.reverse_sort = reverse_sort

    def get_examples(self, num_examples: int, random: bool = True):

        if num_examples > len(self.buffer):
            return []

        elif random:
            return sample(self.buffer, num_examples)
        else:
            return self.buffer[-num_examples:]

    def _add_examples(self, examples: List[ReplayExample]):
        # add examples to buffer and sort based on heuristic
        # remove examples at bottom of list if list exceeds maximum size
        self.buffer.extend(examples)
        self.buffer.sort(key = lambda ex: ex.heuristics_val, reverse = self.reverse_sort)
        
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]  # remove examples with smallest / largest heursitic (depending on if sort is reversed)