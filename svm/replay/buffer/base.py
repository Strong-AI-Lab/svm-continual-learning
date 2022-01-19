from __future__ import annotations

from abc import abstractmethod
from typing import Any, Collection, Dict, List, Tuple, Type, Optional, Union, Callable, TypeVar, TYPE_CHECKING
from random import sample
from copy import deepcopy
from collections import defaultdict

from ..context import ModelPredictionContext

import numpy as np
import torch

if TYPE_CHECKING:
    from ..heuristic.base import Heuristic
    from ..tracker.base import AbstractReplayTracker
    from svm.models.model import ClassificationModel

class ReplayExample():
    def __init__(self, x, y, heuristic: Heuristic, buffer):
        self.x = x
        self.y = y

        self.buffer: AbstractReplayBuffer = buffer  # buffer that contains this replay example
                                                    # (can't type argument due to AbstractReplayBuffer having not been defined yet)
        self.heuristic = heuristic   # heursitic object defining the heuristic for this replay example

    def update_heuristic(self, context: ModelPredictionContext, **kwargs):
        self.heuristic.calculate(context)  # TODO: pass on kwargs

T = TypeVar('T', bound='AbstractReplayTracker')

class AbstractReplayBuffer():

    def __init__(self, heuristic_template: Heuristic, trackers: List[AbstractReplayTracker] = []):

        # store the model that this buffer is attached to
        self.model: ClassificationModel = None

        # heuristics_template is effectively just an instantiated instance of the heuristic type to be used
        # for the replay examples in this bufffer. it is simply used to create copies of it for newly created examples
        self.heuristic_template = heuristic_template
        self.heuristic_template.attach(self)

        self.buffer: Collection
        
        # store trackers in a dictionary format, to allow for simple and fast access
        self.trackers: Dict[Type[AbstractReplayTracker], AbstractReplayTracker] = {}
        for tracker in trackers:
            self.add_tracker(tracker)

    def attach(self, model: ClassificationModel):
        # 'attaches' this replay buffer to the passed model (has no function other than storing its reference, at least for most buffer types)
        self.model = model

    def get_tracker(self, tracker_id: Type[T]) -> T:
        return self.trackers[tracker_id]

    def has_tracker(self, tracker_id: Type[T]) -> bool:
        # checks if a certain tracker has been added to this buffer
        return tracker_id in self.trackers

    def add_tracker(self, tracker: AbstractReplayTracker):
        if type(tracker) in self.trackers:
            # don't allow multiple instances of same tracker type (class) for simplicity. could be solved by giving tracker's unique ids,
            # but this allows for more idiomatic code. also, a single tracker type can be subclassed for different purposes, even if the functionality
            # stays basically the same, as even if nothing is modified using the child class will count as a different class/type (and therefore, different dictionary key)
            # TODO: consider refactoring to use more flexible method? not sure how atm 
            raise ValueError("Can't have more than one instance of the same tracker type tracking the same buffer!")

        self.trackers[type(tracker)] = tracker
        tracker.configure(self)

    @abstractmethod
    def _add_examples(self, examples: List[ReplayExample]):
        # How the examples are added will depend upon the buffer container used (e.g. heap, list, etc.)
        #  - Also, some methods may want to add a 'cut-off' heuristic score or something similar to filter replay examples
        raise NotImplementedError

    @abstractmethod
    def _post_clean_buffer(self, removed_examples: List[ReplayExample]):
        # should be called whenever examples are removed from the replay buffer, by sub-classes
        for tracker in self.trackers.values():
            tracker.post_clean_buffer(removed_examples)
    
    def add_examples(self, context: ModelPredictionContext):

        # 'context' should only contain data relating to the datapoints which are to be added to replay memory
        
        new_examples = []

        for i, x_example in enumerate(context.x):

            context.ex_i = i
            heuristic = self.heuristic_template.copy()
            heuristic.attach(self)

            example = ReplayExample(x_example, context.y_targets[i], heuristic, self)
            example.update_heuristic(context)
            new_examples.append(example)
        
        self._add_examples(new_examples)
        for tracker in self.trackers.values():
            tracker.post_add_examples(new_examples, context)

    def update_examples(self, examples: List[ReplayExample], replay_context: ModelPredictionContext):
        # Updates the passed ReplayExample instances heuristics
        #  - Assumes that examples and replay_context are ordered in the same fashion (i.e. examples[i] corresponds to replay_context.x[i], e.g.)
        for i, example in enumerate(examples):
            replay_context.ex_i = i
            example.update_heuristic(replay_context)

    @abstractmethod
    def get_examples(self, num_examples: int, random: bool = True):
        raise NotImplementedError

    def on_task_switch(self, task_id: Optional[int]):
        # can optionally be implemented to do additional decision making / logic when a task boundary is reached
        # (may never be called depending upon CL setting as well as environment configuration)
        pass