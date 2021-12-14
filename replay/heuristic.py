import torch
from abc import abstractmethod

from .context import SharedStepContext

class Heuristic():

    # Class that represents a calculable heuristic for prioritised replay

    def __init__(self):
        self.val = 0

    @abstractmethod
    def calculate(self, context: SharedStepContext, **kwargs):
        # Calculates and updates the heuristic for the associated replay example
        raise NotImplementedError()

class LossHeuristic():

    # Heuristic that simply reflects the value of the loss function for a given example

    def calculate(self, context: SharedStepContext, **kwargs):
        self.val = torch.mean(context.batch_losses[context.ex_i]).cpu().item()

class SVMBoundaryHeuristic():

    # Heuristic that uses the proximity to the SVM decision boundaries

    def calculate(self, context: SharedStepContext, **kwargs):
        i = context.ex_i
        self.val = 1 / abs(1 - context.y_pred[i][context.y_targets[i]].cpu().item())
