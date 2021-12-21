import torch
from abc import abstractmethod
from typing import List, TYPE_CHECKING, Union
import numpy as np

from ..buffer.base import AbstractReplayBuffer

from ..context import SharedStepContext

class Heuristic():

    # Class that represents a calculable heuristic for prioritised replay

    def __init__(self):
        self.buffer = None
        self.val = 0

    def attach(self, buffer: AbstractReplayBuffer):
        # registers a buffer as this heuristic's container
        self.buffer = buffer

    @abstractmethod
    def calculate(self, context: SharedStepContext, **kwargs):
        # Calculates and updates the heuristic for the associated replay example
        raise NotImplementedError()

    def copy(self, *args, **kwargs) -> 'Heuristic':
        # NOTE: this copy stuff could be a source of error, so this might need looking over at some point
        # returns a copy of this heuristic object
        h_copy = type(self)(*args, **kwargs)
        h_copy.attach(self.buffer)
        return h_copy

class CompoundHeuristic(Heuristic):

    # Heuristic type that combines one or more heuristic values together in some fashion (e.g. adding, multiplying, etc.)

    def __init__(self, children: List[Heuristic]):
        super().__init__()
        self.children = children

    def attach(self, buffer):
        super().attach(buffer)
        for child in self.children:  # propagate attach operation to children
            child.attach(buffer)

    def calculate(self, context: SharedStepContext, **kwargs):
        for heuristic in self.children:
            heuristic.calculate(context)
        self._combine()

    @abstractmethod
    def _combine(self):
        # Abstract method that should describe a way of combining the children heuristic values
        raise NotImplementedError()

    def copy(self, *args, **kwargs):
        new_children = [child.copy() for child in self.children]
        return super().copy(children=new_children, *args, **kwargs)

class ModifierHeuristic(CompoundHeuristic):

    # Specific case of CompoundHeuristic where there should only ever be a single child heuristic

    def __init__(self, child: Heuristic):
        super().__init__([child])
        self.child = child

    def copy(self, *args, **kwargs):
        new_child = self.child.copy()
        return Heuristic.copy(self, new_child)