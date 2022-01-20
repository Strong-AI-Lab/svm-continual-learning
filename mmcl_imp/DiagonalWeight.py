# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:22:15 2021

@author: AURA
"""

from tensorflow.keras.constraints import Constraint 
from tensorflow.keras import backend as K

class DiagonalWeight(Constraint):
    """Constrains the weights to be diagonal.
    """
    def __call__(self, w):
        N = K.int_shape(w)[-1]
        m = K.eye(N)
        w = w * m
        return w
