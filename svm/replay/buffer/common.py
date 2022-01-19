from __future__ import annotations

from abc import abstractmethod
from typing import Any, Collection, Dict, List, Tuple, Type, Optional, Union, Callable, TypeVar, TYPE_CHECKING
from random import sample
from copy import deepcopy
from collections import defaultdict

from svm.replay.heuristic.base import Heuristic
from ..context import ModelPredictionContext
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
            examples_to_remove = self.buffer[:-self.max_buffer_size]
            self.buffer = self.buffer[-self.max_buffer_size:]  # remove examples with smallest / largest heursitic (depending on if sort is reversed)
            self._post_clean_buffer(examples_to_remove)

class ClassSeparatedHeuristicReplayBuffer(AbstractReplayBuffer):

    # Same idea as HeuristicSortedReplayBuffer except we keep a fixed number of examples
    # for each (target) class
    # TODO: refactor replay buffer code so this and HeuristicSortedReplayBuffer can share code

    def __init__(self, heuristic_template: Heuristic, trackers: List[AbstractReplayTracker] = [], class_buffer_size=100, reverse_sort: bool = False):
        super().__init__(heuristic_template, trackers)

        self.buffer: Dict[int, List[ReplayExample]] = defaultdict(lambda: [])
        self.class_buffer_size = class_buffer_size  # number of each class to store
        self.reverse_sort = reverse_sort

    def get_examples(self, num_examples: int):

        buffer_size = sum([len(class_buffer) for class_buffer in self.buffer.values()])

        if num_examples > buffer_size:
            return []

        #TODO: dont sample from current task
        #TODO: add performance metric tracking over time

        class_ids = list(self.buffer.keys())
        sample_class_ids = np.random.choice(class_ids, num_examples)

        ret_examples = []
        for class_id in sample_class_ids:
            ret_examples.append(np.random.choice(self.buffer[class_id]))

        return ret_examples

    def _add_examples(self, examples: List[ReplayExample]):
        # add examples to buffer and sort based on heuristic
        # remove examples at bottom of list if list exceeds maximum size
        for example in examples:
            self.buffer[example.y.cpu().item()].append(example)

        for class_id in self.buffer.keys():
            class_buffer = self.buffer[class_id]
            class_buffer.sort(key = lambda ex: ex.heuristic.val, reverse = self.reverse_sort)

            if len(class_buffer) > self.class_buffer_size:
                examples_to_remove = class_buffer[:-self.class_buffer_size]
                self.buffer[class_id] = class_buffer[-self.class_buffer_size:]  # remove examples with smallest / largest heursitic (depending on if sort is reversed)
                self._post_clean_buffer(examples_to_remove)
        
class DelayedClassSeparatedHeuristicReplayBuffer(ClassSeparatedHeuristicReplayBuffer):

    # Same as ClassSeparatedHeuristicReplayBuffer, except prioritised discarding is delayed until all examples have been seen for a particular class.
    # Can be useful where we want the most accurate classifier possible for a given class before deciding which examples to keep.
    # NOTE: This buffer assumes a CL scenario where a class can only appear in a single task, so can only be used with Class/Task/Domain incremental settings.

    def __init__(self, heuristic_template: Heuristic, trackers: List[AbstractReplayTracker] = [], class_buffer_size=100, reverse_sort: bool = False, update_batch_size: int = 32):
        super().__init__(heuristic_template, trackers, class_buffer_size=class_buffer_size, reverse_sort=reverse_sort)

        self.curr_classes = []  # stores the class ids present in the current task (resets after each task boundary)
        self.update_batch_size = update_batch_size  # the size of the batches used for re-calculating outputs for task examples at end of each task

    def _add_examples(self, examples: List[ReplayExample]):
        for example in examples:
            class_id = example.y.cpu().item()
            if class_id not in self.curr_classes:  # update classes for current task if necessary
                self.curr_classes.append(class_id)

            self.buffer[class_id].append(example)

    def on_task_switch(self, task_id: Optional[int]):
        for class_id in self.curr_classes:
            class_buffer = self.buffer[class_id]

            for i in range(0, len(class_buffer), self.update_batch_size):
                examples = class_buffer[i:i+self.update_batch_size]
                
                x = torch.stack(tuple(ex.x for ex in examples), dim=0)
                y = torch.stack(tuple(ex.y for ex in examples), dim=0)

                y_pred, loss, batch_losses, _ = self.model.shared_step((x, y), None, use_replay=False)  #TODO: hacky solution with setting environment to None, but its currently not used soo....

                context = ModelPredictionContext(-1, x, y, y_pred, batch_losses, loss)
                for i_, ex in enumerate(examples):  # update example heuristics using created context
                    context.ex_i = i_
                    ex.update_heuristic(context)

            class_buffer.sort(key = lambda ex: ex.heuristic.val, reverse = self.reverse_sort)

            if len(class_buffer) > self.class_buffer_size:
                examples_to_remove = class_buffer[:-self.class_buffer_size]
                self.buffer[class_id] = class_buffer[-self.class_buffer_size:]  # remove examples with smallest / largest heursitic (depending on if sort is reversed)
                self._post_clean_buffer(examples_to_remove)

        self.curr_classes = []  # reset curr_classes
