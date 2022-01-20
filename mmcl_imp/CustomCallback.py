# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:57:37 2021

@author: AURA
"""

import tensorflow as tf
import HyperparameterModel
import numpy as np

# Call back function to save the confusion matrix using tensorboard.
class CustomCallback(tf.keras.callbacks.Callback):

    # Save all of your required parameter values in a constructor
    def __init__(self, task_id, hinge_loss, margin_size, kd):
        self.task_id = task_id
        self.hinge_loss = hinge_loss
        self.margin_size = margin_size
        self.kd = kd
        self.hpm = HyperparameterModel.HyperparameterModel()
        self.hpm_weights = ()

    def on_epoch_end(self, epoch, logs):
        # Use the model to predict the values from the validation dataset.
        # only after the first task
        if self.task_id > 0:
            
            params = np.zeros((len(self.hinge_loss), 3))
            for i in range(0, len(self.hinge_loss)):
                params[i][0] = self.hinge_loss[i]
                params[i][1] = self.margin_size[i] * -1 #minus
                params[i][2] = self.kd[i]
            
            self.hpm_weights = self.hpm.model_train(tf.keras.utils.normalize(tf.convert_to_tensor(params)))
            
    def set_values(self, task_id, hinge_loss, margin_size, kd):
        self.task_id = task_id
        self.hinge_loss = hinge_loss
        self.margin_size = margin_size
        self.kd = kd
        
    def return_values(self):
        return self.hpm_weights
