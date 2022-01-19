import torch
from abc import abstractmethod
from typing import List
import numpy as np

from ..context import ModelPredictionContext
from .base import Heuristic, CompoundHeuristic, ModifierHeuristic

class LossHeuristic(Heuristic):

    # Heuristic that simply reflects the value of the loss function for a given example

    def _calculate(self, context: ModelPredictionContext, **kwargs):
        self.val = torch.mean(context.batch_losses[context.ex_i]).cpu().item()

class ProductHeuristic(CompoundHeuristic):

    # Calculates the product of all the children heuristics

    def _combine(self):
        self.val = np.prod([h.val for h in self.children])

class InversionHeuristic(ModifierHeuristic):

    # Calculates the inverse of the child heuristic
    def _combine(self):
        c_val = self.child.val if self.child.val != 0 else 0.0000001  # prevent division by zero
        self.val = 1 / c_val  

class ExponentiationHeuristic(ModifierHeuristic):

    def __init__(self, child: Heuristic, n: int):
        super().__init__(child)
        self.n = n  # power to raise heuristic value to

    def _combine(self):
        self.val = pow(self.child.val, self.n)

    def copy(self, *args, **kwargs):
        return super().copy(*args, n=self.n, **kwargs)

class SquaringHeuristic(ExponentiationHeuristic):

    # Special, common case of Exponentation heuristic where n is 2

    def __init__(self, child: Heuristic, should_update: bool=True):
        super().__init__(child, 2, should_update=should_update)

class WeightedSummationHeuristic(CompoundHeuristic):

    def __init__(self, children: List[Heuristic], coeffs: List[float] = [], should_update: bool=True):
        # coefffs is a list of float coefficients that each child heuristic value is multiplied by (pairwise)
        # to allow different weightings for each heuristic
        super().__init__(children, should_update=should_update)
        self.coeffs = coeffs or [1 for _ in children]  # default to 1 for each heuristic (equally weighted)

    def _combine(self):
        self.val = np.sum([child_h.val * self.coeffs[i] for i, child_h in enumerate(self.children)])

    def copy(self, *args, **kwargs):
        return super().copy(*args, coeffs=list(self.coeffs), **kwargs)