import torch
from abc import abstractmethod
from typing import List
import numpy as np

from ..context import ModelPredictionContext
from .base import Heuristic
from ..buffer.base import AbstractReplayBuffer
from ..tracker.common import ClassDistributionTracker

class SVMBoundaryHeuristic(Heuristic):

    # Heuristic that uses the proximity to the SVM decision boundaries

    def __init__(self, offset: float = 0.0, should_update: bool = True):
        super().__init__(should_update=should_update)
        self.offset = offset  # defines an offset w.r.t the positive decision boundary, to be used as the 'boundary' to calculate proximity to

    def _calculate(self, context: ModelPredictionContext, **kwargs):
        i = context.ex_i
        self.val = abs((1 + self.offset) - context.y_pred[i][context.y_targets[i]].cpu().item())

class ClassRepresentationHeuristic(Heuristic):

    # Heuristic that gives examples weightings according to how under-represented they are in the replay buffer

    def _calculate(self, context: ModelPredictionContext, **kwargs):

        try:
            class_tracker = self.buffer.get_tracker(ClassDistributionTracker)
        except KeyError as e:
            raise KeyError("ClassRepresentationHeuristic requires a ClassDistributionTracker to calculate its heuristic value") from e

        class_distribution = class_tracker.get_distribution()

        num_examples = np.sum(class_distribution)

        i = context.ex_i
        ex_class = context.y_targets[i]
        self.val = 1 / (max(class_distribution[ex_class], 1) / max(num_examples, 1))  # use max function in case no examples exist


