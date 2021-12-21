from __future__ import annotations

from abc import abstractmethod
from typing import Any, Collection, Dict, List, Tuple, Type, Optional, Union, Callable, TypeVar, TYPE_CHECKING

from ..heuristic.base import Heuristic
from ..context import SharedStepContext

import numpy as np
import torch

if TYPE_CHECKING:
    from ..buffer.base import AbstractReplayBuffer, ReplayExample

class AbstractReplayTracker():

    # Class used for tracking replay buffer characteristics
    #  - could range from tracking class distributions, to average heuristic values, etc.
    
    def __init__(self, name='default'):
        self.name = name  # used to differentiate instances of the same type of tracker
        self.buffer: AbstractReplayBuffer  # Buffer that we are tracking, set in the configure method

    def configure(self, buffer: AbstractReplayBuffer):
        # called when the tracker has been added to a replay buffer, allowing for any needed one time configuration operations
        self.buffer = buffer

    @abstractmethod
    def post_add_examples(self, examples: List[ReplayExample], context: SharedStepContext):
        # Hook that is called by the replay buffer class after examples have been added to the buffer
        pass

    @abstractmethod
    def post_get_examples(self, examples: List[ReplayExample], context: SharedStepContext):
        # Hook that is called by the replay buffer class after examples have been retrieved from the buffer
        pass

    @abstractmethod
    def post_clean_buffer(self, examples_removed: List[ReplayExample]):
        # Hook that is called by the replay buffer class after the buffer has been 'cleaned' (undesirable examples removed)
        pass


## NOTE: the stuff below can really just be implemented by the tracker class itself, as it has a reference to the buffer it belongs to,
##       so i've removed it for now. while this could cause code duplication for simpler trackers, it shouldn't be a big issue  


# class MetaReplayTracker(AbstractReplayTracker):

#     # Tracker that tracks characteristics of another replay tracker, belonging to the same replay buffer
#     #  - Primary use case is to enable flexible logging / tracking of tracker statistics to establish trends etc.

#     def __init__(self, trackers: List[AbstractReplayTracker], name='default'):
#         super().__init__(name=name)

#         # keeps track of the replay trackers that we are tracking
#         self.trackers = trackers

# class CallbackMetaReplayTracker(MetaReplayTracker):

#     # simple implementation of a MetaReplayTracker where we just want to invoke a callback on each replay tracker that we are tracking,
#     # while storing a value for each invocation
#     #  - callback could include chained if statements to accommodate more than one tracker, as well as having variables for more advanced tracking

#     # flags used to describe the nature of the tracker callback's invocation
#     ADD_EXAMPLES_FLAG = 0
#     GET_EXAMPLES_FLAG = 1

#     def __init__(self, trackers: List[AbstractReplayTracker], tracker_callback: Callable[[AbstractReplayTracker, int, MetaReplayTracker], Any]):
#         super().__init__(trackers)

#         # callback function that is called in the post_add_examples and post_get_examples methods
#         #  - is passed the tracker object that we are tracking, as well as a flag describing whether this was called due to a add/get examples operation (see flags)
#         #  - is also passed this meta replay tracker object as well
#         #  - if the returned value is None, this is ommitted from the register. this can be used to save space / computation with, e.g., sampling at set intervals
#         #    (e.g. every 10 invocations return an actual value)
#         self.tracker_callback = tracker_callback

#         # values returned from tracker_callback are stored in lists, organised such that each tracked replay tracker has its own register
#         self.tracker_registers = {}
#         for tracker in trackers:
#             self.tracker_registers[tracker_id] = []

#     def post_add_examples(self, examples: List[ReplayExample], context: SharedStepContext):
#         for tracker_id in self.tracker_ids:
#             tracker = self.buffer.get_tracker(tracker_id)
#             self.tracker_registers[tracker_id].append(self.tracker_callback(tracker, CallbackMetaReplayTracker.ADD_EXAMPLES_FLAG))

#     def post_get_examples(self, examples: List[ReplayExample], context: SharedStepContext):
#         for tracker_id in self.tracker_ids:
#             tracker = self.buffer.get_tracker(tracker_id)
#             callback_val = self.tracker_callback(tracker, CallbackMetaReplayTracker.GET_EXAMPLES_FLAG)
            
#             if callback_val is not None:
#                 self.tracker_registers[tracker_id].append(callback_val)