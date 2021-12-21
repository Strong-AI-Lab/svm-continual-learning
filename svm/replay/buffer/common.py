from __future__ import annotations

from abc import abstractmethod
from typing import Any, Collection, Dict, List, Tuple, Type, Optional, Union, Callable, TypeVar, TYPE_CHECKING
from random import sample
from copy import deepcopy
from collections import defaultdict

from svm.replay.heuristic.base import Heuristic
from ..context import SharedStepContext
from .base import AbstractReplayBuffer, ReplayExample

import numpy as np
import torch

if TYPE_CHECKING:
    from ..tracker.base import AbstractReplayTracker

class NonFunctionalReplayBuffer(AbstractReplayBuffer):

    # "Implementation" of replay buffer that does not do anything

    def __init__(self):
        super().__init__(Heuristic())

    def add_examples(self, examples: List[ReplayExample]):
        pass

    def get_examples(self, num_examples: int, random: bool = True):
        return []

class HeuristicSortedReplayBuffer(AbstractReplayBuffer):

    # Replay buffer where the replay examples are sorted by some heuristic
    #  - Replay examples that have higher / lower heuristic (depending on if reverse sort or not)
    #    will be in memory for more time, thus they will influence model dynamics more.

    def __init__(self, heuristic_template: Heuristic, trackers: List[AbstractReplayTracker] = [], max_buffer_size: int = 1_000, reverse_sort: bool = False):
        super().__init__(heuristic_template, trackers)

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
            examples_to_remove = self.buffer[:len(self.buffer) - self.max_buffer_size]
            self.buffer = self.buffer[-self.max_buffer_size:]  # remove examples with smallest / largest heursitic (depending on if sort is reversed)
            self._post_clean_buffer(examples_to_remove)