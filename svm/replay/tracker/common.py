from __future__ import annotations

from typing import Any, Collection, Dict, List, Tuple, Type, Optional, Union, Callable, TypeVar, TYPE_CHECKING

from ..heuristic.base import Heuristic
from ..context import ModelPredictionContext

import numpy as np
import torch

from .base import AbstractReplayTracker

if TYPE_CHECKING:
    from ..buffer.base import ReplayExample

class ClassDistributionTracker(AbstractReplayTracker):

    # Tracker used to track the distribution of classes within the replay buffer

    def __init__(self, n_classes: int, onehot: bool = False):
        super().__init__()
        
        self.class_counts: List[int] = [0 for _ in range(n_classes)]
        self.prev_counts: List[List[int]] = []  # stores previous counts

        self.onehot = onehot  # whether the target outputs are defined as one-hot vector encodings or not (class ID)

    def _process_example(self, example: ReplayExample, dir: int):
        try:
            y = torch.nonzero(example.y).item() if self.onehot else example.y
        except ValueError as e:
            raise ValueError('Output targets were specified as one-hot vector encodings, yet more than one index is non-zero') from e

        self.class_counts[y] += dir

    def post_add_examples(self, examples: List[ReplayExample], context: ModelPredictionContext):
        # only called once per timestep
        for example in examples:
            self._process_example(example, 1)
        self.prev_counts.append(list(self.class_counts))  # store copy of current counts

    def post_clean_buffer(self, examples_removed: List[ReplayExample]):
        for example in examples_removed:
            self._process_example(example, -1)
        # don't store current counts here as this method could be called any amount of times per timestep

    def get_distribution(self):
        return self.class_counts

    def get_history(self):
        return self.prev_counts