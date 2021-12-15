
#import torch

from abc import abstractmethod
from typing import Collection, Dict, List, Tuple, Type, Optional, Union
from random import sample
from copy import deepcopy

from .heuristic import Heuristic
from .context import SharedStepContext

import numpy as np

class ReplayExample():
    def __init__(self, x, y, heuristic: Heuristic):
        self.x = x
        self.y = y

        self.heuristic = heuristic   # heursitic object defining the heuristic for this replay example

    def update_heuristic(self, context: SharedStepContext, **kwargs):
        self.heuristic.calculate(context)  # TODO: pass on kwargs

class AbstractReplayBuffer():

    def __init__(self, heuristic_template: Heuristic):
        self.buffer: Collection
        # heuristics_template is effectively just an instantiated instance of the heuristic type to be used
        # for the replay examples in this bufffer. it is simply used to create copies of it for newly created examples
        self.heuristic_template = heuristic_template

    @abstractmethod
    def _add_examples(self, examples: List[ReplayExample]):
        # How the examples are added will depend upon the buffer container used (e.g. heap, list, etc.)
        #  - Also, some methods may want to add a 'cut-off' heuristic score or something similar to filter replay examples
        raise NotImplementedError
    
    def add_examples(self, context: SharedStepContext):
        
        new_examples = []

        for i, x_example in enumerate(context.x):

            context.ex_i = i
            heuristic = deepcopy(self.heuristic_template)

            example = ReplayExample(x_example, context.y_targets[i], heuristic)
            example.update_heuristic(context)
            new_examples.append(example)
        
        self._add_examples(new_examples)

    def update_examples(self, examples: List[ReplayExample], replay_context: SharedStepContext):
        # Updates the passed ReplayExample instances heuristics
        #  - Assumes that examples and replay_context are ordered in the same fashion (i.e. examples[i] corresponds to replay_context.x[i], e.g.)
        for i, example in enumerate(examples):
            replay_context.ex_i = i
            example.update_heuristic(replay_context)

    @abstractmethod
    def get_examples(self, num_examples: int, random: bool = True):
        raise NotImplementedError

class NonFunctionalReplayBuffer(AbstractReplayBuffer):

    # "Implementation" of replay buffer that does not do anything

    def __init__(self):
        super().__init__(None)

    def add_examples(self, examples: List[ReplayExample]):
        pass

    def get_examples(self, num_examples: int, random: bool = True):
        return []

class HeuristicSortedReplayBuffer(AbstractReplayBuffer):

    # Replay buffer where the replay examples are sorted by some heuristic
    #  - Replay examples that have higher / lower heuristic (depending on if reverse sort or not)
    #    will be in memory for more time, thus they will influence model dynamics more.

    def __init__(self, heuristic_template: Heuristic, max_buffer_size: int = 1_000, reverse_sort: bool = False):
        super().__init__(heuristic_template)

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
        self.buffer.sort(key = lambda ex: ex.heuristic.val, reverse = self.reverse_sort)
        
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]  # remove examples with smallest / largest heursitic (depending on if sort is reversed)