import torch
from abc import abstractmethod
from typing import List
import numpy as np

from ..context import SharedStepContext
from .base import Heuristic
from ..buffer.base import AbstractReplayBuffer
from ..tracker.common import ClassDistributionTracker

class SVMBoundaryHeuristic(Heuristic):

    # Heuristic that uses the proximity to the SVM decision boundaries

    def calculate(self, context: SharedStepContext, **kwargs):
        i = context.ex_i
        self.val = abs(1 - context.y_pred[i][context.y_targets[i]].cpu().item())

class ClassRepresentationHeuristic(Heuristic):

    # Heuristic that gives examples weightings according to how under-represented they are in the replay buffer

    def calculate(self, context: SharedStepContext, **kwargs):

        try:
            class_tracker = self.buffer.get_tracker(ClassDistributionTracker)
        except KeyError as e:
            raise KeyError("ClassEqualisingHeuristic requires a ClassDistributionTracker to calculate its heuristic value") from e

        class_distribution = class_tracker.get_distribution()

        num_examples = np.sum(class_distribution)

        i = context.ex_i
        ex_class = context.y_targets[i]
        self.val = 1 / (max(class_distribution[ex_class], 1) / max(num_examples, 1))  # use max function in case no examples exist


