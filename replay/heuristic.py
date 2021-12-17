import torch
from abc import abstractmethod
from typing import List
import numpy as np

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
        self.val = abs(1 - context.y_pred[i][context.y_targets[i]].cpu().item())

class CompoundHeuristic(Heuristic):

    # Heuristic type that combines one or more heuristic values together in some fashion (e.g. adding, multiplying, etc.)

    def __init__(self, children: List[Heuristic]):
        super().__init__()
        self.children = children

    def calculate(self, context: SharedStepContext, **kwargs):
        for heuristic in self.children:
            heuristic.calculate(context)
        self._combine()

    @abstractmethod
    def _combine(self):
        # Abstract method that should describe a way of combining the children heuristic values
        raise NotImplementedError()

class ModifierHeuristic(CompoundHeuristic):

    # Specific case of CompoundHeuristic where there should only ever be a single child heuristic

    def __init__(self, child: Heuristic):
        super().__init__([child])
        self.child = child

class WeightedSummationHeuristic(CompoundHeuristic):

    def __init__(self, children: List[Heuristic], coeffs: List[float] = []):
        # coefffs is a list of float coefficients that each child heuristic value is multiplied by (pairwise)
        # to allow different weightings for each heuristic
        super().__init__(children)
        self.coeffs = coeffs or [1 for _ in children]  # default to 1 for each heuristic (equally weighted)

    def _combine(self):
        self.val = np.sum([child_h.val * self.coeffs[i] for i, child_h in enumerate(self.children)])

class InversionHeuristic(ModifierHeuristic):

    # Calculates the inverse of the child heuristic
    def _combine(self):
        self.val = 1 / self.child.val

class ExponentiationHeuristic(ModifierHeuristic):

    def __init__(self, child: Heuristic, n: int):
        super().__init__(child)
        self.n = n  # power to raise heuristic value to

    def _combine(self):
        self.val = pow(self.child.val, self.n)

class SquaringHeuristic(ExponentiationHeuristic):

    # Special, common case of Exponentation heuristic where n is 2

    def __init__(self, child: Heuristic):
        super().__init__(child, 2)
