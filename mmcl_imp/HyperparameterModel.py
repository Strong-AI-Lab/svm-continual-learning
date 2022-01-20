# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:02:23 2021

@author: Diana
"""

import numpy as np
from copy import deepcopy
from IPython import display
import pandas as pd
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.chdir('C:/Users/AURA/Desktop/AISTATS2021/method')
import tensorflow as tf
import DiagonalWeight

class HyperparameterModel:

    def __init__(self):

        self.x = tf.keras.layers.Input(shape = [3])
        
        self.layer1 = tf.keras.layers.Dense(3, kernel_constraint = DiagonalWeight.DiagonalWeight(), use_bias = False)(self.x)
        self.output = tf.keras.layers.Dense(1, name = "output", use_bias = False, kernel_constraint = tf.keras.constraints.NonNeg())(self.layer1)
        
        self.model = tf.keras.models.Model(inputs = self.x, outputs = self.output)
        
    def model_train(self, x):
        self.x = x
        self.y = tf.zeros((x.shape[0]))
        
        self.model.compile(tf.keras.optimizers.SGD(learning_rate=0.001), 
                          loss = 'mean_squared_error', 
                          metrics = ['mse'], run_eagerly = True)
            
        self.model.fit(self.x, self.y, epochs = 10, verbose = 0)
        
        return tuple(self.model.layers[len(self.model.layers) - 1].get_weights()[0][:,0].tolist())

#x = tf.convert_to_tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
#hp = HyperparameterModel()
#hp.model_train(x)

