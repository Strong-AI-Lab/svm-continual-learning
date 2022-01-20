# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:07:49 2020

@author: Diana
"""

import numpy as np
from copy import deepcopy
#import matplotlib.pyplot as plt
from IPython import display
import pandas as pd
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import DiagonalWeight #for output layer
import HyperparameterModel #for hyperparameter learning
import CustomCallback
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from numpy import linalg as la

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def input_variable(shape):
    initial = tf.random.truncated_normal(shape=shape, mean=0.0, stddev=0.0)
    return tf.Variable(initial)


class Model:

    def __init__(self, dataset, in_dim, out_dim, batch_size, train_mode = [0], net_mode = "single", rows_data = 28, cols_data = 28, channels_data = 1, reinit_layers = False, checkpointing = False, keep_sv = 0, type_sv = "ranking"):
        self.dataset = dataset
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        #self.x = x # This is a placeholder that'll be used to feed data
        #self.y_observed = y_
        self.err = 0.5
        self.err_oneclass = 0
        #set rho
        self.rho = tf.ones([self.out_dim])
        self.rho_oneclass = tf.zeros([1])
        self.train_mode = train_mode
        #anchor examples
        self.anchors_x = []
        self.anchors_y = []
        self.anchors = []
        self.penalty_parameter = tf.constant(0.5, dtype = tf.float32)
        self.batch_id = 0
        self.task_id = 0
        self.net_mode = net_mode
        self.reinit_layers = reinit_layers
        self.rows_data = rows_data
        self.cols_data = cols_data
        self.channels_data = channels_data
        self.checkpointing = checkpointing
        self.predictions_current_all = []
        self.predictions_current = []
        self.predictions_new = []
        #self.mean_scores = np.zeros((self.out_dim, self.out_dim)) #row is a task, column is the score for 
        #self.sd_scores = np.zeros((self.out_dim, self.out_dim))
        self.hinge_loss_perexample = 0.0
        self.margin_size_perexample = 0.0
        self.kd_perexample = 0.0
        #callback
        self.callback = CustomCallback.CustomCallback(0, 0.0, 0.0, 0.0)
        #percentange of training examples to keep as SVs
        self.keep_sv = keep_sv
        #type sv
        self.type_sv = type_sv
        
        #alpha, beta, gamma
        self.nu_parameter = 0.5
        self.alpha = 0.5 #hinge
        self.beta = 0.8 #margin size change
        self.gamma = 0.8 #knowledge distillation
        
        #callback to restore model to a previous state
        #self.model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.getcwd() + "\\checkpoints\\cp.ckpt", 
                                                            #verbose=1, 
                                                            #save_weights_only=True,
                                                            #save_freq=2)
        
        #self.x = input_variable([self.batch_size, self.in_dim])
        #self.y_observed = input_variable([batch_size, self.out_dim])
        
        #single model for all tasks
        
        #self.model = tf.keras.models.Sequential([
                    #tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(self.rows_data, self.cols_data, self.channels_data)),
                    #tf.keras.layers.MaxPooling2D((2, 2)), 
                    #tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
                    #tf.keras.layers.MaxPooling2D((2, 2)),
                    #tf.keras.layers.Flatten(),
                    #tf.keras.layers.Dense(1024, activation = 'relu'),
                    #tf.keras.layers.Dropout(.2),
                    #tf.keras.layers.Dense(self.out_dim, name = "output")])
        
        #self.y = self.model.get_layer('output').output
        
        
        #input for models with Functional API
        self.x_input = tf.keras.layers.Input(shape = [self.rows_data, self.cols_data, self.channels_data]) 
        
        #input for one-class model with Functional API
        #self.x_input_oneclass = self.x_input

        
        #single model for all tasks svm        
        #self.single_svm_first_conv = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(self.rows_data, self.cols_data, self.channels_data))(self.x_input)
        #self.single_svm_first_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.single_svm_first_conv) 
        #self.single_svm_second_conv = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')(self.single_svm_first_pool)
        #self.single_svm_second_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.single_svm_second_conv)
        #self.single_svm_flatten = tf.keras.layers.Flatten()(self.single_svm_second_pool)
        #self.single_svm_dense = tf.keras.layers.Dense(1024, activation = 'relu')(self.single_svm_flatten)
        #self.single_svm_dropout = tf.keras.layers.Dropout(.2)(self.single_svm_dense)
        #self.single_svm_readout = tf.keras.layers.Dense(self.out_dim, name = "readout")(self.single_svm_dropout)
        #self.single_svm_output = tf.keras.layers.Dense(self.out_dim, name = "output")(self.single_svm_dropout)
        
        #self.svm_model = tf.keras.models.Model(inputs = self.x_input, outputs = self.single_svm_output)
        
        
        #multiple headers model, for regular classication (softmax)

        
        
        if self.dataset == "mnist":
            self.cnn_mnist()
            
        elif self.dataset == "cifar10":
            self.cnn_cifar10()
        
        #one class subnetwork
        # #multiple headers model, for svm classication (C or nu)
        #self.svm_oneclass_first_shared_conv = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=[self.rows_data, self.cols_data, self.channels_data], padding="same")(self.x_input_oneclass)
        #self.svm_oneclass_first_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.svm_oneclass_first_shared_conv)
        #self.svm_oneclass_second_shared_conv = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="same")(self.svm_oneclass_first_shared_pool)
        #self.svm_oneclass_second_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.svm_oneclass_second_shared_conv)
        #per task layers (10) 
        #first a convolutional layer
        #self.svm_oneclass_first_ind_conv_task0 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task0",)(self.svm_oneclass_second_shared_pool)
        #self.svm_oneclass_first_ind_pool_task0 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task0')(self.svm_oneclass_first_ind_conv_task0)
        #then flatten
        #self.svm_oneclass_first_ind_flatten_task0 = tf.keras.layers.Flatten(name = "first_ind_flatten_task0")(self.svm_oneclass_first_ind_pool_task0)
        #now dense
        #self.svm_oneclass_first_ind_dense_task0 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task0')(self.svm_oneclass_first_ind_flatten_task0)
        #now dropout
        #self.svm_oneclass_first_ind_dropout_task0 = tf.keras.layers.Dropout(0.2)(self.svm_oneclass_first_ind_dense_task0)
        #single outputs
        #self.svm_oneclass_first_ind_oneclass_output = tf.keras.layers.Dense(1, name = "output", use_bias = False)(self.svm_oneclass_first_ind_dropout_task0)
        
        #output of the one class
        #self.svm_oneclass_model = tf.keras.models.Model(inputs = self.x_input, outputs = self.svm_oneclass_first_ind_oneclass_output)
        #self.svm_oneclass_model.save_weights('./checkpoints/oneclass_model')
        
        #self.list_oneclass_models = []
        
        #a single model per task, for binary classication per task
        #single outputs
        # self.ind_output_task0 = tf.keras.layers.Dense(1, name = "ind_output_task0", activation = "sigmoid")(self.first_ind_dropout_task0)
        # self.ind_output_task1 = tf.keras.layers.Dense(1, name = "ind_output_task1", activation = "sigmoid")(self.first_ind_dropout_task1)
        # self.ind_output_task2 = tf.keras.layers.Dense(1, name = "ind_output_task2", activation = "sigmoid")(self.first_ind_dropout_task2)
        # self.ind_output_task3 = tf.keras.layers.Dense(1, name = "ind_output_task3", activation = "sigmoid")(self.first_ind_dropout_task3)
        # self.ind_output_task4 = tf.keras.layers.Dense(1, name = "ind_output_task4", activation = "sigmoid")(self.first_ind_dropout_task4)
        # self.ind_output_task5 = tf.keras.layers.Dense(1, name = "ind_output_task5", activation = "sigmoid")(self.first_ind_dropout_task5)
        # self.ind_output_task6 = tf.keras.layers.Dense(1, name = "ind_output_task6", activation = "sigmoid")(self.first_ind_dropout_task6)
        # self.ind_output_task7 = tf.keras.layers.Dense(1, name = "ind_output_task7", activation = "sigmoid")(self.first_ind_dropout_task7)
        # self.ind_output_task8 = tf.keras.layers.Dense(1, name = "ind_output_task8", activation = "sigmoid")(self.first_ind_dropout_task8)
        # self.ind_output_task9 = tf.keras.layers.Dense(1, name = "ind_output_task9", activation = "sigmoid")(self.first_ind_dropout_task9)
        
        # self.model_task0 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task0)
        # self.model_task1 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task1)
        # self.model_task2 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task2)
        # self.model_task3 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task3)
        # self.model_task4 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task4)
        # self.model_task5 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task5)
        # self.model_task6 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task6)
        # self.model_task7 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task7)
        # self.model_task8 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task8)
        # self.model_task9 = tf.keras.models.Model(inputs = self.x_input, outputs = self.ind_output_task9)
        
        # self.models = [self.model_task0, self.model_task1, self.model_task2, self.model_task3, self.model_task4, 
        #                self.model_task5, self.model_task6, self.model_task7, self.model_task8, self.model_task9]
        
        
        # #new CNN
        # # First convolutional layer
        # self.first_conv_weight = weight_variable([5, 5, 1, 32])
        # self.first_conv_bias = bias_variable([32])

        # self.input_image = tf.reshape(self.x, [-1, 28, 28, 1])

        # self.first_conv_activation = tf.nn.relu(
        #     tf.nn.conv2d(input = self.input_image, filter = self.first_conv_weight, padding='SAME') + self.first_conv_bias
        #     )
        # self.first_conv_pool = tf.nn.max_pool(self.first_conv_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # # Second convolutional layer
        # self.second_conv_weight = weight_variable([5, 5, 32, 64])
        # self.second_conv_bias = bias_variable([64])

        # self.second_conv_activation = tf.nn.relu(
        #     tf.nn.conv2d(input = self.first_conv_pool, filter = self.second_conv_weight, padding='SAME') + self.second_conv_bias
        # )
        # self.second_conv_pool = tf.nn.max_pool(self.second_conv_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # # Fully-connected layer (Dense Layer)
        # self.dense_layer_weight = weight_variable([7 * 7 * 64, 1024])
        # self.dense_layer_bias = bias_variable([1024])

        # self.second_conv_pool_flatten = tf.reshape(self.second_conv_pool, [-1, 7 * 7 * 64])
        # self.dense_layer_activation = tf.nn.relu(
        #     tf.matmul(self.second_conv_pool_flatten, self.dense_layer_weight)
        #         + self.dense_layer_bias
        #     )

        # # Dropout, to avoid over-fitting
        # self.keep_prob = tf.constant(0.5)
        # self.h_fc1_drop = tf.nn.dropout(self.dense_layer_activation, self.keep_prob)

        # # Readout layer
        # self.readout_weight = weight_variable([1024, self.out_dim])
        # self.readout_bias = bias_variable([self.out_dim])

        # self.y = tf.matmul(self.h_fc1_drop, self.readout_weight) + self.readout_bias

    def cnn_mnist(self):
        # #multiple headers model, for svm classication (C or nu)
        if self.train_mode == [1000]:
            self.svm_first_shared_conv = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=[self.rows_data, self.cols_data, self.channels_data], padding="same")(self.x_input)
            self.svm_first_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.svm_first_shared_conv)
            self.svm_second_shared_conv = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="same")(self.svm_first_shared_pool)
            self.svm_second_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.svm_second_shared_conv)
            #per task layers (10) 
            #first a convolutional layer
            self.svm_first_ind_conv_task0 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task0",)(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task0 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task0')(self.svm_first_ind_conv_task0)
            self.svm_first_ind_conv_task1 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task1")(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task1 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task1')(self.svm_first_ind_conv_task1)
            self.svm_first_ind_conv_task2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task2")(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task2 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task2')(self.svm_first_ind_conv_task2)
            self.svm_first_ind_conv_task3 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task3")(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task3 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task3')(self.svm_first_ind_conv_task3)
            self.svm_first_ind_conv_task4 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task4")(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task4 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task4')(self.svm_first_ind_conv_task4)
            self.svm_first_ind_conv_task5 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task5")(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task5 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task5')(self.svm_first_ind_conv_task5)
            self.svm_first_ind_conv_task6 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task6")(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task6 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task6')(self.svm_first_ind_conv_task6)
            self.svm_first_ind_conv_task7 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task7")(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task7 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task7')(self.svm_first_ind_conv_task7)
            self.svm_first_ind_conv_task8 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task8")(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task8 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task8')(self.svm_first_ind_conv_task8)
            self.svm_first_ind_conv_task9 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task9")(self.svm_second_shared_pool)
            self.svm_first_ind_pool_task9 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task9')(self.svm_first_ind_conv_task9)
            #then flatten
            self.svm_first_ind_flatten_task0 = tf.keras.layers.Flatten(name = "first_ind_flatten_task0")(self.svm_first_ind_pool_task0)
            self.svm_first_ind_flatten_task1 = tf.keras.layers.Flatten(name = "first_ind_flatten_task1")(self.svm_first_ind_pool_task1)
            self.svm_first_ind_flatten_task2 = tf.keras.layers.Flatten(name = "first_ind_flatten_task2")(self.svm_first_ind_pool_task2)
            self.svm_first_ind_flatten_task3 = tf.keras.layers.Flatten(name = "first_ind_flatten_task3")(self.svm_first_ind_pool_task3)
            self.svm_first_ind_flatten_task4 = tf.keras.layers.Flatten(name = "first_ind_flatten_task4")(self.svm_first_ind_pool_task4)
            self.svm_first_ind_flatten_task5 = tf.keras.layers.Flatten(name = "first_ind_flatten_task5")(self.svm_first_ind_pool_task5)
            self.svm_first_ind_flatten_task6 = tf.keras.layers.Flatten(name = "first_ind_flatten_task6")(self.svm_first_ind_pool_task6)
            self.svm_first_ind_flatten_task7 = tf.keras.layers.Flatten(name = "first_ind_flatten_task7")(self.svm_first_ind_pool_task7)
            self.svm_first_ind_flatten_task8 = tf.keras.layers.Flatten(name = "first_ind_flatten_task8")(self.svm_first_ind_pool_task8)
            self.svm_first_ind_flatten_task9 = tf.keras.layers.Flatten(name = "first_ind_flatten_task9")(self.svm_first_ind_pool_task9)
            #now dense
            self.svm_first_ind_dense_task0 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task0')(self.svm_first_ind_flatten_task0)
            self.svm_first_ind_dense_task1 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task1')(self.svm_first_ind_flatten_task1)
            self.svm_first_ind_dense_task2 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task2')(self.svm_first_ind_flatten_task2)
            self.svm_first_ind_dense_task3 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task3')(self.svm_first_ind_flatten_task3)
            self.svm_first_ind_dense_task4 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task4')(self.svm_first_ind_flatten_task4)
            self.svm_first_ind_dense_task5 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task5')(self.svm_first_ind_flatten_task5)
            self.svm_first_ind_dense_task6 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task6')(self.svm_first_ind_flatten_task6)
            self.svm_first_ind_dense_task7 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task7')(self.svm_first_ind_flatten_task7)
            self.svm_first_ind_dense_task8 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task8')(self.svm_first_ind_flatten_task8)
            self.svm_first_ind_dense_task9 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task9')(self.svm_first_ind_flatten_task9)
            #now dropout
            self.svm_first_ind_dropout_task0 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task0)
            self.svm_first_ind_dropout_task1 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task1)
            self.svm_first_ind_dropout_task2 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task2)
            self.svm_first_ind_dropout_task3 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task3)
            self.svm_first_ind_dropout_task4 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task4)
            self.svm_first_ind_dropout_task5 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task5)
            self.svm_first_ind_dropout_task6 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task6)
            self.svm_first_ind_dropout_task7 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task7)
            self.svm_first_ind_dropout_task8 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task8)
            self.svm_first_ind_dropout_task9 = tf.keras.layers.Dropout(0.2)(self.svm_first_ind_dense_task9)        
            #single outputs
            self.svm_ind_output_task0 = tf.keras.layers.Dense(1, name = "ind_output_task0")(self.svm_first_ind_dropout_task0)
            self.svm_ind_output_task1 = tf.keras.layers.Dense(1, name = "ind_output_task1")(self.svm_first_ind_dropout_task1)
            self.svm_ind_output_task2 = tf.keras.layers.Dense(1, name = "ind_output_task2")(self.svm_first_ind_dropout_task2)
            self.svm_ind_output_task3 = tf.keras.layers.Dense(1, name = "ind_output_task3")(self.svm_first_ind_dropout_task3)   
            self.svm_ind_output_task4 = tf.keras.layers.Dense(1, name = "ind_output_task4")(self.svm_first_ind_dropout_task4)   
            self.svm_ind_output_task5 = tf.keras.layers.Dense(1, name = "ind_output_task5")(self.svm_first_ind_dropout_task5)   
            self.svm_ind_output_task6 = tf.keras.layers.Dense(1, name = "ind_output_task6")(self.svm_first_ind_dropout_task6)   
            self.svm_ind_output_task7 = tf.keras.layers.Dense(1, name = "ind_output_task7")(self.svm_first_ind_dropout_task7)   
            self.svm_ind_output_task8 = tf.keras.layers.Dense(1, name = "ind_output_task8")(self.svm_first_ind_dropout_task8)   
            self.svm_ind_output_task9 = tf.keras.layers.Dense(1, name = "ind_output_task9")(self.svm_first_ind_dropout_task9)   
            #now combine
            self.svm_ind_output_combined = tf.keras.layers.Concatenate(axis = 1)([self.svm_ind_output_task0,
                                                      self.svm_ind_output_task1,
                                                      self.svm_ind_output_task2,
                                                      self.svm_ind_output_task3,
                                                      self.svm_ind_output_task4,
                                                      self.svm_ind_output_task5,
                                                      self.svm_ind_output_task6,
                                                      self.svm_ind_output_task7,
                                                      self.svm_ind_output_task8,
                                                      self.svm_ind_output_task9])
            #tf.keras.layers.Dropout(.2),
            self.svm_last_shared_output = tf.keras.layers.Dense(self.out_dim, name = "output", kernel_constraint = DiagonalWeight.DiagonalWeight())(self.svm_ind_output_combined)
            #self.svm_last_shared_output = SparseLayer.SparseLayer(self.out_dim, name = "output")(self.svm_ind_output_combined)
            
            self.svm_model_multiple = tf.keras.models.Model(inputs = self.x_input, outputs = self.svm_last_shared_output)
            self.svm_model_multiple.save_weights('./checkpoints/svm_model_multiple')
            
            
            # previous multiple headers model, for svm classication (C or nu)
            self.prev_svm_first_shared_conv = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=[self.rows_data, self.cols_data, self.channels_data], padding="same")(self.x_input)
            self.prev_svm_first_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.prev_svm_first_shared_conv)
            self.prev_svm_second_shared_conv = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="same")(self.prev_svm_first_shared_pool)
            self.prev_svm_second_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.prev_svm_second_shared_conv)
            #per task layers (10) 
            #first a convolutional layer
            self.prev_svm_first_ind_conv_task0 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task0",)(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task0 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task0')(self.prev_svm_first_ind_conv_task0)
            self.prev_svm_first_ind_conv_task1 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task1")(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task1 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task1')(self.prev_svm_first_ind_conv_task1)
            self.prev_svm_first_ind_conv_task2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task2")(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task2 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task2')(self.prev_svm_first_ind_conv_task2)
            self.prev_svm_first_ind_conv_task3 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task3")(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task3 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task3')(self.prev_svm_first_ind_conv_task3)
            self.prev_svm_first_ind_conv_task4 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task4")(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task4 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task4')(self.prev_svm_first_ind_conv_task4)
            self.prev_svm_first_ind_conv_task5 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task5")(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task5 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task5')(self.prev_svm_first_ind_conv_task5)
            self.prev_svm_first_ind_conv_task6 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task6")(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task6 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task6')(self.prev_svm_first_ind_conv_task6)
            self.prev_svm_first_ind_conv_task7 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task7")(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task7 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task7')(self.prev_svm_first_ind_conv_task7)
            self.prev_svm_first_ind_conv_task8 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task8")(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task8 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task8')(self.prev_svm_first_ind_conv_task8)
            self.prev_svm_first_ind_conv_task9 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "pre_first_ind_conv_task9")(self.prev_svm_second_shared_pool)
            self.prev_svm_first_ind_pool_task9 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task9')(self.prev_svm_first_ind_conv_task9)
            #then flatten
            self.prev_svm_first_ind_flatten_task0 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task0")(self.prev_svm_first_ind_pool_task0)
            self.prev_svm_first_ind_flatten_task1 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task1")(self.prev_svm_first_ind_pool_task1)
            self.prev_svm_first_ind_flatten_task2 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task2")(self.prev_svm_first_ind_pool_task2)
            self.prev_svm_first_ind_flatten_task3 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task3")(self.prev_svm_first_ind_pool_task3)
            self.prev_svm_first_ind_flatten_task4 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task4")(self.prev_svm_first_ind_pool_task4)
            self.prev_svm_first_ind_flatten_task5 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task5")(self.prev_svm_first_ind_pool_task5)
            self.prev_svm_first_ind_flatten_task6 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task6")(self.prev_svm_first_ind_pool_task6)
            self.prev_svm_first_ind_flatten_task7 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task7")(self.prev_svm_first_ind_pool_task7)
            self.prev_svm_first_ind_flatten_task8 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task8")(self.prev_svm_first_ind_pool_task8)
            self.prev_svm_first_ind_flatten_task9 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task9")(self.prev_svm_first_ind_pool_task9)
            #now dense
            self.prev_svm_first_ind_dense_task0 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task0')(self.prev_svm_first_ind_flatten_task0)
            self.prev_svm_first_ind_dense_task1 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task1')(self.prev_svm_first_ind_flatten_task1)
            self.prev_svm_first_ind_dense_task2 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task2')(self.prev_svm_first_ind_flatten_task2)
            self.prev_svm_first_ind_dense_task3 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task3')(self.prev_svm_first_ind_flatten_task3)
            self.prev_svm_first_ind_dense_task4 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task4')(self.prev_svm_first_ind_flatten_task4)
            self.prev_svm_first_ind_dense_task5 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task5')(self.prev_svm_first_ind_flatten_task5)
            self.prev_svm_first_ind_dense_task6 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task6')(self.prev_svm_first_ind_flatten_task6)
            self.prev_svm_first_ind_dense_task7 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task7')(self.prev_svm_first_ind_flatten_task7)
            self.prev_svm_first_ind_dense_task8 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task8')(self.prev_svm_first_ind_flatten_task8)
            self.prev_svm_first_ind_dense_task9 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task9')(self.prev_svm_first_ind_flatten_task9)
            #now dropout
            self.prev_svm_first_ind_dropout_task0 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task0)
            self.prev_svm_first_ind_dropout_task1 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task1)
            self.prev_svm_first_ind_dropout_task2 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task2)
            self.prev_svm_first_ind_dropout_task3 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task3)
            self.prev_svm_first_ind_dropout_task4 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task4)
            self.prev_svm_first_ind_dropout_task5 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task5)
            self.prev_svm_first_ind_dropout_task6 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task6)
            self.prev_svm_first_ind_dropout_task7 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task7)
            self.prev_svm_first_ind_dropout_task8 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task8)
            self.prev_svm_first_ind_dropout_task9 = tf.keras.layers.Dropout(0.2)(self.prev_svm_first_ind_dense_task9)        
            #single outputs
            self.prev_svm_ind_output_task0 = tf.keras.layers.Dense(1, name = "pre_ind_output_task0")(self.prev_svm_first_ind_dropout_task0)
            self.prev_svm_ind_output_task1 = tf.keras.layers.Dense(1, name = "pre_ind_output_task1")(self.prev_svm_first_ind_dropout_task1)
            self.prev_svm_ind_output_task2 = tf.keras.layers.Dense(1, name = "pre_ind_output_task2")(self.prev_svm_first_ind_dropout_task2)
            self.prev_svm_ind_output_task3 = tf.keras.layers.Dense(1, name = "pre_ind_output_task3")(self.prev_svm_first_ind_dropout_task3)   
            self.prev_svm_ind_output_task4 = tf.keras.layers.Dense(1, name = "pre_ind_output_task4")(self.prev_svm_first_ind_dropout_task4)   
            self.prev_svm_ind_output_task5 = tf.keras.layers.Dense(1, name = "pre_ind_output_task5")(self.prev_svm_first_ind_dropout_task5)   
            self.prev_svm_ind_output_task6 = tf.keras.layers.Dense(1, name = "pre_ind_output_task6")(self.prev_svm_first_ind_dropout_task6)   
            self.prev_svm_ind_output_task7 = tf.keras.layers.Dense(1, name = "pre_ind_output_task7")(self.prev_svm_first_ind_dropout_task7)   
            self.prev_svm_ind_output_task8 = tf.keras.layers.Dense(1, name = "pre_ind_output_task8")(self.prev_svm_first_ind_dropout_task8)   
            self.prev_svm_ind_output_task9 = tf.keras.layers.Dense(1, name = "pre_ind_output_task9")(self.prev_svm_first_ind_dropout_task9)   
            #now combine
            self.prev_svm_ind_output_combined = tf.keras.layers.Concatenate(axis = 1)([self.prev_svm_ind_output_task0,
                                                      self.prev_svm_ind_output_task1,
                                                      self.prev_svm_ind_output_task2,
                                                      self.prev_svm_ind_output_task3,
                                                      self.prev_svm_ind_output_task4,
                                                      self.prev_svm_ind_output_task5,
                                                      self.prev_svm_ind_output_task6,
                                                      self.prev_svm_ind_output_task7,
                                                      self.prev_svm_ind_output_task8,
                                                      self.prev_svm_ind_output_task9])
            #tf.keras.layers.Dropout(.2),
            self.prev_svm_last_shared_output = tf.keras.layers.Dense(self.out_dim, name = "pre_output")(self.prev_svm_ind_output_combined)        
            
            self.previous_svm_model_multiple = tf.keras.models.Model(inputs = self.x_input, outputs = self.prev_svm_last_shared_output)
            
        if self.train_mode == [0]:
            self.first_shared_conv = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=[self.rows_data, self.cols_data, self.channels_data], padding="same")(self.x_input)
            self.first_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.first_shared_conv)
            self.second_shared_conv = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="same")(self.first_shared_pool)
            self.second_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.second_shared_conv)
            #per task layers (10) 
            #first a convolutional layer
            self.first_ind_conv_task0 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task0",)(self.second_shared_pool)
            self.first_ind_pool_task0 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task0')(self.first_ind_conv_task0)
            self.first_ind_conv_task1 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task1")(self.second_shared_pool)
            self.first_ind_pool_task1 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task1')(self.first_ind_conv_task1)
            self.first_ind_conv_task2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task2")(self.second_shared_pool)
            self.first_ind_pool_task2 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task2')(self.first_ind_conv_task2)
            self.first_ind_conv_task3 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task3")(self.second_shared_pool)
            self.first_ind_pool_task3 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task3')(self.first_ind_conv_task3)
            self.first_ind_conv_task4 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task4")(self.second_shared_pool)
            self.first_ind_pool_task4 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task4')(self.first_ind_conv_task4)
            self.first_ind_conv_task5 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task5")(self.second_shared_pool)
            self.first_ind_pool_task5 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task5')(self.first_ind_conv_task5)
            self.first_ind_conv_task6 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task6")(self.second_shared_pool)
            self.first_ind_pool_task6 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task6')(self.first_ind_conv_task6)
            self.first_ind_conv_task7 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task7")(self.second_shared_pool)
            self.first_ind_pool_task7 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task7')(self.first_ind_conv_task7)
            self.first_ind_conv_task8 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task8")(self.second_shared_pool)
            self.first_ind_pool_task8 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task8')(self.first_ind_conv_task8)
            self.first_ind_conv_task9 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = "first_ind_conv_task9")(self.second_shared_pool)
            self.first_ind_pool_task9 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task9')(self.first_ind_conv_task9)
            #then flatten
            self.first_ind_flatten_task0 = tf.keras.layers.Flatten(name = "first_ind_flatten_task0")(self.first_ind_pool_task0)
            self.first_ind_flatten_task1 = tf.keras.layers.Flatten(name = "first_ind_flatten_task1")(self.first_ind_pool_task1)
            self.first_ind_flatten_task2 = tf.keras.layers.Flatten(name = "first_ind_flatten_task2")(self.first_ind_pool_task2)
            self.first_ind_flatten_task3 = tf.keras.layers.Flatten(name = "first_ind_flatten_task3")(self.first_ind_pool_task3)
            self.first_ind_flatten_task4 = tf.keras.layers.Flatten(name = "first_ind_flatten_task4")(self.first_ind_pool_task4)
            self.first_ind_flatten_task5 = tf.keras.layers.Flatten(name = "first_ind_flatten_task5")(self.first_ind_pool_task5)
            self.first_ind_flatten_task6 = tf.keras.layers.Flatten(name = "first_ind_flatten_task6")(self.first_ind_pool_task6)
            self.first_ind_flatten_task7 = tf.keras.layers.Flatten(name = "first_ind_flatten_task7")(self.first_ind_pool_task7)
            self.first_ind_flatten_task8 = tf.keras.layers.Flatten(name = "first_ind_flatten_task8")(self.first_ind_pool_task8)
            self.first_ind_flatten_task9 = tf.keras.layers.Flatten(name = "first_ind_flatten_task9")(self.first_ind_pool_task9)
            #now dense
            self.first_ind_dense_task0 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task0')(self.first_ind_flatten_task0)
            self.first_ind_dense_task1 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task1')(self.first_ind_flatten_task1)
            self.first_ind_dense_task2 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task2')(self.first_ind_flatten_task2)
            self.first_ind_dense_task3 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task3')(self.first_ind_flatten_task3)
            self.first_ind_dense_task4 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task4')(self.first_ind_flatten_task4)
            self.first_ind_dense_task5 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task5')(self.first_ind_flatten_task5)
            self.first_ind_dense_task6 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task6')(self.first_ind_flatten_task6)
            self.first_ind_dense_task7 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task7')(self.first_ind_flatten_task7)
            self.first_ind_dense_task8 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task8')(self.first_ind_flatten_task8)
            self.first_ind_dense_task9 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task9')(self.first_ind_flatten_task9)
            #now dropout
            self.first_ind_dropout_task0 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task0)
            self.first_ind_dropout_task1 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task1)
            self.first_ind_dropout_task2 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task2)
            self.first_ind_dropout_task3 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task3)
            self.first_ind_dropout_task4 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task4)
            self.first_ind_dropout_task5 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task5)
            self.first_ind_dropout_task6 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task6)
            self.first_ind_dropout_task7 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task7)
            self.first_ind_dropout_task8 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task8)
            self.first_ind_dropout_task9 = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task9)
            #single outputs
            self.ind_output_task0 = tf.keras.layers.Dense(1, name = "ind_output_task0")(self.first_ind_dropout_task0)
            self.ind_output_task1 = tf.keras.layers.Dense(1, name = "ind_output_task1")(self.first_ind_dropout_task1)
            self.ind_output_task2 = tf.keras.layers.Dense(1, name = "ind_output_task2")(self.first_ind_dropout_task2)
            self.ind_output_task3 = tf.keras.layers.Dense(1, name = "ind_output_task3")(self.first_ind_dropout_task3)
            self.ind_output_task4 = tf.keras.layers.Dense(1, name = "ind_output_task4")(self.first_ind_dropout_task4)
            self.ind_output_task5 = tf.keras.layers.Dense(1, name = "ind_output_task5")(self.first_ind_dropout_task5)
            self.ind_output_task6 = tf.keras.layers.Dense(1, name = "ind_output_task6")(self.first_ind_dropout_task6)
            self.ind_output_task7 = tf.keras.layers.Dense(1, name = "ind_output_task7")(self.first_ind_dropout_task7)
            self.ind_output_task8 = tf.keras.layers.Dense(1, name = "ind_output_task8")(self.first_ind_dropout_task8)
            self.ind_output_task9 = tf.keras.layers.Dense(1, name = "ind_output_task9")(self.first_ind_dropout_task9)
            #now combine
            self.ind_output_combined = tf.keras.layers.Concatenate(axis = 1)([self.ind_output_task0,
                                                                              self.ind_output_task1,
                                                                              self.ind_output_task2,
                                                                              self.ind_output_task3,
                                                                              self.ind_output_task4,
                                                                              self.ind_output_task5,
                                                                              self.ind_output_task6,
                                                                              self.ind_output_task7,
                                                                              self.ind_output_task8,
                                                                              self.ind_output_task9])
            self.last_shared_output = tf.keras.layers.Dense(self.out_dim, name = "output")(self.ind_output_combined)
            
            self.model_multiple = tf.keras.models.Model(inputs = self.x_input, outputs = self.last_shared_output)
        


    def cnn_cifar10(self):
        # #multiple headers model, for svm classication (C or nu)
        if self.train_mode == [1000] or self.train_mode == [100]:
            #example https://gist.github.com/josefelixsandoval/fad43016c9bca96ea529a2ff4340eba3
            self.svm_first_shared_conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[self.rows_data, self.cols_data, self.channels_data], padding="same")(self.x_input)
            self.svm_second_shared_conv = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(self.svm_first_shared_conv)
            self.svm_first_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.svm_second_shared_conv)
            self.svm_first_dropout = tf.keras.layers.Dropout(0.25)(self.svm_first_shared_pool)
            
            self.svm_third_shared_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(self.svm_first_dropout)
            self.svm_fourth_shared_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")(self.svm_third_shared_conv)
            self.svm_second_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.svm_fourth_shared_conv)
            self.svm_second_dropout = tf.keras.layers.Dropout(0.25)(self.svm_second_shared_pool)
            #per task layers (10) 
            #first a convolutional layer
            self.svm_first_ind_conv_task0 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task0",)(self.svm_second_dropout)
            self.svm_first_ind_pool_task0 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task0')(self.svm_first_ind_conv_task0)
            self.svm_first_ind_conv_task1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task1")(self.svm_second_dropout)
            self.svm_first_ind_pool_task1 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task1')(self.svm_first_ind_conv_task1)
            self.svm_first_ind_conv_task2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task2")(self.svm_second_dropout)
            self.svm_first_ind_pool_task2 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task2')(self.svm_first_ind_conv_task2)
            self.svm_first_ind_conv_task3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task3")(self.svm_second_dropout)
            self.svm_first_ind_pool_task3 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task3')(self.svm_first_ind_conv_task3)
            self.svm_first_ind_conv_task4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task4")(self.svm_second_dropout)
            self.svm_first_ind_pool_task4 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task4')(self.svm_first_ind_conv_task4)
            self.svm_first_ind_conv_task5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task5")(self.svm_second_dropout)
            self.svm_first_ind_pool_task5 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task5')(self.svm_first_ind_conv_task5)
            self.svm_first_ind_conv_task6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task6")(self.svm_second_dropout)
            self.svm_first_ind_pool_task6 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task6')(self.svm_first_ind_conv_task6)
            self.svm_first_ind_conv_task7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task7")(self.svm_second_dropout)
            self.svm_first_ind_pool_task7 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task7')(self.svm_first_ind_conv_task7)
            self.svm_first_ind_conv_task8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task8")(self.svm_second_dropout)
            self.svm_first_ind_pool_task8 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task8')(self.svm_first_ind_conv_task8)
            self.svm_first_ind_conv_task9 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task9")(self.svm_second_dropout)
            self.svm_first_ind_pool_task9 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task9')(self.svm_first_ind_conv_task9)
            #then flatten
            self.svm_first_ind_flatten_task0 = tf.keras.layers.Flatten(name = "first_ind_flatten_task0")(self.svm_first_ind_pool_task0)
            self.svm_first_ind_flatten_task1 = tf.keras.layers.Flatten(name = "first_ind_flatten_task1")(self.svm_first_ind_pool_task1)
            self.svm_first_ind_flatten_task2 = tf.keras.layers.Flatten(name = "first_ind_flatten_task2")(self.svm_first_ind_pool_task2)
            self.svm_first_ind_flatten_task3 = tf.keras.layers.Flatten(name = "first_ind_flatten_task3")(self.svm_first_ind_pool_task3)
            self.svm_first_ind_flatten_task4 = tf.keras.layers.Flatten(name = "first_ind_flatten_task4")(self.svm_first_ind_pool_task4)
            self.svm_first_ind_flatten_task5 = tf.keras.layers.Flatten(name = "first_ind_flatten_task5")(self.svm_first_ind_pool_task5)
            self.svm_first_ind_flatten_task6 = tf.keras.layers.Flatten(name = "first_ind_flatten_task6")(self.svm_first_ind_pool_task6)
            self.svm_first_ind_flatten_task7 = tf.keras.layers.Flatten(name = "first_ind_flatten_task7")(self.svm_first_ind_pool_task7)
            self.svm_first_ind_flatten_task8 = tf.keras.layers.Flatten(name = "first_ind_flatten_task8")(self.svm_first_ind_pool_task8)
            self.svm_first_ind_flatten_task9 = tf.keras.layers.Flatten(name = "first_ind_flatten_task9")(self.svm_first_ind_pool_task9)
            #now dense
            self.svm_first_ind_dense_task0 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task0')(self.svm_first_ind_flatten_task0)
            self.svm_first_ind_dense_task1 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task1')(self.svm_first_ind_flatten_task1)
            self.svm_first_ind_dense_task2 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task2')(self.svm_first_ind_flatten_task2)
            self.svm_first_ind_dense_task3 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task3')(self.svm_first_ind_flatten_task3)
            self.svm_first_ind_dense_task4 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task4')(self.svm_first_ind_flatten_task4)
            self.svm_first_ind_dense_task5 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task5')(self.svm_first_ind_flatten_task5)
            self.svm_first_ind_dense_task6 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task6')(self.svm_first_ind_flatten_task6)
            self.svm_first_ind_dense_task7 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task7')(self.svm_first_ind_flatten_task7)
            self.svm_first_ind_dense_task8 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task8')(self.svm_first_ind_flatten_task8)
            self.svm_first_ind_dense_task9 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task9')(self.svm_first_ind_flatten_task9)
            #now dropout
            self.svm_first_ind_dropout_task0 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task0)
            self.svm_first_ind_dropout_task1 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task1)
            self.svm_first_ind_dropout_task2 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task2)
            self.svm_first_ind_dropout_task3 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task3)
            self.svm_first_ind_dropout_task4 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task4)
            self.svm_first_ind_dropout_task5 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task5)
            self.svm_first_ind_dropout_task6 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task6)
            self.svm_first_ind_dropout_task7 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task7)
            self.svm_first_ind_dropout_task8 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task8)
            self.svm_first_ind_dropout_task9 = tf.keras.layers.Dropout(0.5)(self.svm_first_ind_dense_task9)        
            #single outputs
            self.svm_ind_output_task0 = tf.keras.layers.Dense(1, name = "ind_output_task0")(self.svm_first_ind_dropout_task0)
            self.svm_ind_output_task1 = tf.keras.layers.Dense(1, name = "ind_output_task1")(self.svm_first_ind_dropout_task1)
            self.svm_ind_output_task2 = tf.keras.layers.Dense(1, name = "ind_output_task2")(self.svm_first_ind_dropout_task2)
            self.svm_ind_output_task3 = tf.keras.layers.Dense(1, name = "ind_output_task3")(self.svm_first_ind_dropout_task3)   
            self.svm_ind_output_task4 = tf.keras.layers.Dense(1, name = "ind_output_task4")(self.svm_first_ind_dropout_task4)   
            self.svm_ind_output_task5 = tf.keras.layers.Dense(1, name = "ind_output_task5")(self.svm_first_ind_dropout_task5)   
            self.svm_ind_output_task6 = tf.keras.layers.Dense(1, name = "ind_output_task6")(self.svm_first_ind_dropout_task6)   
            self.svm_ind_output_task7 = tf.keras.layers.Dense(1, name = "ind_output_task7")(self.svm_first_ind_dropout_task7)   
            self.svm_ind_output_task8 = tf.keras.layers.Dense(1, name = "ind_output_task8")(self.svm_first_ind_dropout_task8)   
            self.svm_ind_output_task9 = tf.keras.layers.Dense(1, name = "ind_output_task9")(self.svm_first_ind_dropout_task9)   
            #now combine
            self.svm_ind_output_combined = tf.keras.layers.Concatenate(axis = 1)([self.svm_ind_output_task0,
                                                      self.svm_ind_output_task1,
                                                      self.svm_ind_output_task2,
                                                      self.svm_ind_output_task3,
                                                      self.svm_ind_output_task4,
                                                      self.svm_ind_output_task5,
                                                      self.svm_ind_output_task6,
                                                      self.svm_ind_output_task7,
                                                      self.svm_ind_output_task8,
                                                      self.svm_ind_output_task9])
            #tf.keras.layers.Dropout(.2),
            self.svm_last_shared_output = tf.keras.layers.Dense(self.out_dim, name = "output", kernel_constraint = DiagonalWeight.DiagonalWeight())(self.svm_ind_output_combined)
            #self.svm_last_shared_output = SparseLayer.SparseLayer(self.out_dim, name = "output")(self.svm_ind_output_combined)
            
            self.svm_model_multiple = tf.keras.models.Model(inputs = self.x_input, outputs = self.svm_last_shared_output)
            self.svm_model_multiple.save_weights('./checkpoints/svm_model_multiple')
            
            
            # previous multiple headers model, for svm classication (C or nu)
            self.prev_svm_first_shared_conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[self.rows_data, self.cols_data, self.channels_data], padding="same")(self.x_input)
            self.prev_svm_second_shared_conv = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(self.prev_svm_first_shared_conv)
            self.prev_svm_first_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.prev_svm_second_shared_conv)
            self.prev_svm_first_dropout = tf.keras.layers.Dropout(0.25)(self.prev_svm_first_shared_pool)
            
            self.prev_svm_third_shared_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(self.prev_svm_first_dropout)
            self.prev_svm_fourth_shared_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")(self.prev_svm_third_shared_conv)
            self.prev_svm_second_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.prev_svm_fourth_shared_conv)
            self.prev_svm_second_dropout = tf.keras.layers.Dropout(0.25)(self.prev_svm_second_shared_pool)
            
            #per task layers (10) 
            #first a convolutional layer
            self.prev_svm_first_ind_conv_task0 = tf.keras.layers.Conv2D(128, (3,3 ), activation='relu', name = "pre_first_ind_conv_task0",)(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task0 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task0')(self.prev_svm_first_ind_conv_task0)
            self.prev_svm_first_ind_conv_task1 = tf.keras.layers.Conv2D(128, (3,3 ), activation='relu', name = "pre_first_ind_conv_task1")(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task1 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task1')(self.prev_svm_first_ind_conv_task1)
            self.prev_svm_first_ind_conv_task2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "pre_first_ind_conv_task2")(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task2 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task2')(self.prev_svm_first_ind_conv_task2)
            self.prev_svm_first_ind_conv_task3 = tf.keras.layers.Conv2D(128, (3,3 ), activation='relu', name = "pre_first_ind_conv_task3")(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task3 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task3')(self.prev_svm_first_ind_conv_task3)
            self.prev_svm_first_ind_conv_task4 = tf.keras.layers.Conv2D(128, (3,3 ), activation='relu', name = "pre_first_ind_conv_task4")(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task4 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task4')(self.prev_svm_first_ind_conv_task4)
            self.prev_svm_first_ind_conv_task5 = tf.keras.layers.Conv2D(128, (3,3 ), activation='relu', name = "pre_first_ind_conv_task5")(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task5 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task5')(self.prev_svm_first_ind_conv_task5)
            self.prev_svm_first_ind_conv_task6 = tf.keras.layers.Conv2D(128, (3,3 ), activation='relu', name = "pre_first_ind_conv_task6")(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task6 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task6')(self.prev_svm_first_ind_conv_task6)
            self.prev_svm_first_ind_conv_task7 = tf.keras.layers.Conv2D(128, (3,3 ), activation='relu', name = "pre_first_ind_conv_task7")(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task7 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task7')(self.prev_svm_first_ind_conv_task7)
            self.prev_svm_first_ind_conv_task8 = tf.keras.layers.Conv2D(128, (3,3 ), activation='relu', name = "pre_first_ind_conv_task8")(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task8 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task8')(self.prev_svm_first_ind_conv_task8)
            self.prev_svm_first_ind_conv_task9 = tf.keras.layers.Conv2D(128, (3,3 ), activation='relu', name = "pre_first_ind_conv_task9")(self.prev_svm_second_dropout)
            self.prev_svm_first_ind_pool_task9 = tf.keras.layers.MaxPooling2D((2, 2), name = 'pre_first_ind_pool_task9')(self.prev_svm_first_ind_conv_task9)
            #then flatten
            self.prev_svm_first_ind_flatten_task0 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task0")(self.prev_svm_first_ind_pool_task0)
            self.prev_svm_first_ind_flatten_task1 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task1")(self.prev_svm_first_ind_pool_task1)
            self.prev_svm_first_ind_flatten_task2 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task2")(self.prev_svm_first_ind_pool_task2)
            self.prev_svm_first_ind_flatten_task3 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task3")(self.prev_svm_first_ind_pool_task3)
            self.prev_svm_first_ind_flatten_task4 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task4")(self.prev_svm_first_ind_pool_task4)
            self.prev_svm_first_ind_flatten_task5 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task5")(self.prev_svm_first_ind_pool_task5)
            self.prev_svm_first_ind_flatten_task6 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task6")(self.prev_svm_first_ind_pool_task6)
            self.prev_svm_first_ind_flatten_task7 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task7")(self.prev_svm_first_ind_pool_task7)
            self.prev_svm_first_ind_flatten_task8 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task8")(self.prev_svm_first_ind_pool_task8)
            self.prev_svm_first_ind_flatten_task9 = tf.keras.layers.Flatten(name = "pre_first_ind_flatten_task9")(self.prev_svm_first_ind_pool_task9)
            #now dense
            self.prev_svm_first_ind_dense_task0 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task0')(self.prev_svm_first_ind_flatten_task0)
            self.prev_svm_first_ind_dense_task1 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task1')(self.prev_svm_first_ind_flatten_task1)
            self.prev_svm_first_ind_dense_task2 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task2')(self.prev_svm_first_ind_flatten_task2)
            self.prev_svm_first_ind_dense_task3 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task3')(self.prev_svm_first_ind_flatten_task3)
            self.prev_svm_first_ind_dense_task4 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task4')(self.prev_svm_first_ind_flatten_task4)
            self.prev_svm_first_ind_dense_task5 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task5')(self.prev_svm_first_ind_flatten_task5)
            self.prev_svm_first_ind_dense_task6 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task6')(self.prev_svm_first_ind_flatten_task6)
            self.prev_svm_first_ind_dense_task7 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task7')(self.prev_svm_first_ind_flatten_task7)
            self.prev_svm_first_ind_dense_task8 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task8')(self.prev_svm_first_ind_flatten_task8)
            self.prev_svm_first_ind_dense_task9 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'pre_first_ind_dense_task9')(self.prev_svm_first_ind_flatten_task9)
            #now dropout
            self.prev_svm_first_ind_dropout_task0 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task0)
            self.prev_svm_first_ind_dropout_task1 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task1)
            self.prev_svm_first_ind_dropout_task2 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task2)
            self.prev_svm_first_ind_dropout_task3 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task3)
            self.prev_svm_first_ind_dropout_task4 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task4)
            self.prev_svm_first_ind_dropout_task5 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task5)
            self.prev_svm_first_ind_dropout_task6 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task6)
            self.prev_svm_first_ind_dropout_task7 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task7)
            self.prev_svm_first_ind_dropout_task8 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task8)
            self.prev_svm_first_ind_dropout_task9 = tf.keras.layers.Dropout(0.5)(self.prev_svm_first_ind_dense_task9)        
            #single outputs
            self.prev_svm_ind_output_task0 = tf.keras.layers.Dense(1, name = "pre_ind_output_task0")(self.prev_svm_first_ind_dropout_task0)
            self.prev_svm_ind_output_task1 = tf.keras.layers.Dense(1, name = "pre_ind_output_task1")(self.prev_svm_first_ind_dropout_task1)
            self.prev_svm_ind_output_task2 = tf.keras.layers.Dense(1, name = "pre_ind_output_task2")(self.prev_svm_first_ind_dropout_task2)
            self.prev_svm_ind_output_task3 = tf.keras.layers.Dense(1, name = "pre_ind_output_task3")(self.prev_svm_first_ind_dropout_task3)   
            self.prev_svm_ind_output_task4 = tf.keras.layers.Dense(1, name = "pre_ind_output_task4")(self.prev_svm_first_ind_dropout_task4)   
            self.prev_svm_ind_output_task5 = tf.keras.layers.Dense(1, name = "pre_ind_output_task5")(self.prev_svm_first_ind_dropout_task5)   
            self.prev_svm_ind_output_task6 = tf.keras.layers.Dense(1, name = "pre_ind_output_task6")(self.prev_svm_first_ind_dropout_task6)   
            self.prev_svm_ind_output_task7 = tf.keras.layers.Dense(1, name = "pre_ind_output_task7")(self.prev_svm_first_ind_dropout_task7)   
            self.prev_svm_ind_output_task8 = tf.keras.layers.Dense(1, name = "pre_ind_output_task8")(self.prev_svm_first_ind_dropout_task8)   
            self.prev_svm_ind_output_task9 = tf.keras.layers.Dense(1, name = "pre_ind_output_task9")(self.prev_svm_first_ind_dropout_task9)   
            #now combine
            self.prev_svm_ind_output_combined = tf.keras.layers.Concatenate(axis = 1)([self.prev_svm_ind_output_task0,
                                                      self.prev_svm_ind_output_task1,
                                                      self.prev_svm_ind_output_task2,
                                                      self.prev_svm_ind_output_task3,
                                                      self.prev_svm_ind_output_task4,
                                                      self.prev_svm_ind_output_task5,
                                                      self.prev_svm_ind_output_task6,
                                                      self.prev_svm_ind_output_task7,
                                                      self.prev_svm_ind_output_task8,
                                                      self.prev_svm_ind_output_task9])
            #tf.keras.layers.Dropout(.2),
            self.prev_svm_last_shared_output = tf.keras.layers.Dense(self.out_dim, name = "output", kernel_constraint = DiagonalWeight.DiagonalWeight())(self.prev_svm_ind_output_combined)
            
            self.previous_svm_model_multiple = tf.keras.models.Model(inputs = self.x_input, outputs = self.prev_svm_last_shared_output)
            
        if self.train_mode == [0]:
            #multiple headers model, for regular classication (softmax)
           
            self.first_shared_conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[self.rows_data, self.cols_data, self.channels_data], padding="same")(self.x_input)
            self.second_shared_conv = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(self.first_shared_conv)
            self.first_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.second_shared_conv)
            self.first_shared_dropout = tf.keras.layers.Dropout(0.25)(self.first_shared_pool)
            
            self.third_shared_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(self.first_shared_dropout)
            self.fourth_shared_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")(self.third_shared_conv)
            self.second_shared_pool = tf.keras.layers.MaxPooling2D((2, 2))(self.fourth_shared_conv)
            self.second_shared_dropout = tf.keras.layers.Dropout(0.25)(self.second_shared_pool)
            #per task layers (10) 
            #first a convolutional layer
            self.first_ind_conv_task0 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task0",)(self.second_shared_dropout)
            self.first_ind_pool_task0 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task0')(self.first_ind_conv_task0)
            self.first_ind_conv_task1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task1")(self.second_shared_dropout)
            self.first_ind_pool_task1 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task1')(self.first_ind_conv_task1)
            self.first_ind_conv_task2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task2")(self.second_shared_dropout)
            self.first_ind_pool_task2 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task2')(self.first_ind_conv_task2)
            self.first_ind_conv_task3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task3")(self.second_shared_dropout)
            self.first_ind_pool_task3 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task3')(self.first_ind_conv_task3)
            self.first_ind_conv_task4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task4")(self.second_shared_dropout)
            self.first_ind_pool_task4 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task4')(self.first_ind_conv_task4)
            self.first_ind_conv_task5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task5")(self.second_shared_dropout)
            self.first_ind_pool_task5 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task5')(self.first_ind_conv_task5)
            self.first_ind_conv_task6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task6")(self.second_shared_dropout)
            self.first_ind_pool_task6 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task6')(self.first_ind_conv_task6)
            self.first_ind_conv_task7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task7")(self.second_shared_dropout)
            self.first_ind_pool_task7 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task7')(self.first_ind_conv_task7)
            self.first_ind_conv_task8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task8")(self.second_shared_dropout)
            self.first_ind_pool_task8 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task8')(self.first_ind_conv_task8)
            self.first_ind_conv_task9 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name = "first_ind_conv_task9")(self.second_shared_dropout)
            self.first_ind_pool_task9 = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task9')(self.first_ind_conv_task9)
            #then flatten
            self.first_ind_flatten_task0 = tf.keras.layers.Flatten(name = "first_ind_flatten_task0")(self.first_ind_pool_task0)
            self.first_ind_flatten_task1 = tf.keras.layers.Flatten(name = "first_ind_flatten_task1")(self.first_ind_pool_task1)
            self.first_ind_flatten_task2 = tf.keras.layers.Flatten(name = "first_ind_flatten_task2")(self.first_ind_pool_task2)
            self.first_ind_flatten_task3 = tf.keras.layers.Flatten(name = "first_ind_flatten_task3")(self.first_ind_pool_task3)
            self.first_ind_flatten_task4 = tf.keras.layers.Flatten(name = "first_ind_flatten_task4")(self.first_ind_pool_task4)
            self.first_ind_flatten_task5 = tf.keras.layers.Flatten(name = "first_ind_flatten_task5")(self.first_ind_pool_task5)
            self.first_ind_flatten_task6 = tf.keras.layers.Flatten(name = "first_ind_flatten_task6")(self.first_ind_pool_task6)
            self.first_ind_flatten_task7 = tf.keras.layers.Flatten(name = "first_ind_flatten_task7")(self.first_ind_pool_task7)
            self.first_ind_flatten_task8 = tf.keras.layers.Flatten(name = "first_ind_flatten_task8")(self.first_ind_pool_task8)
            self.first_ind_flatten_task9 = tf.keras.layers.Flatten(name = "first_ind_flatten_task9")(self.first_ind_pool_task9)
            #now dense
            self.first_ind_dense_task0 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task0')(self.first_ind_flatten_task0)
            self.first_ind_dense_task1 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task1')(self.first_ind_flatten_task1)
            self.first_ind_dense_task2 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task2')(self.first_ind_flatten_task2)
            self.first_ind_dense_task3 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task3')(self.first_ind_flatten_task3)
            self.first_ind_dense_task4 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task4')(self.first_ind_flatten_task4)
            self.first_ind_dense_task5 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task5')(self.first_ind_flatten_task5)
            self.first_ind_dense_task6 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task6')(self.first_ind_flatten_task6)
            self.first_ind_dense_task7 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task7')(self.first_ind_flatten_task7)
            self.first_ind_dense_task8 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task8')(self.first_ind_flatten_task8)
            self.first_ind_dense_task9 = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task9')(self.first_ind_flatten_task9)
            #now dropout
            self.first_ind_dropout_task0 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task0)
            self.first_ind_dropout_task1 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task1)
            self.first_ind_dropout_task2 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task2)
            self.first_ind_dropout_task3 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task3)
            self.first_ind_dropout_task4 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task4)
            self.first_ind_dropout_task5 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task5)
            self.first_ind_dropout_task6 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task6)
            self.first_ind_dropout_task7 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task7)
            self.first_ind_dropout_task8 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task8)
            self.first_ind_dropout_task9 = tf.keras.layers.Dropout(0.5)(self.first_ind_dense_task9)
            #single outputs
            self.ind_output_task0 = tf.keras.layers.Dense(1, name = "ind_output_task0")(self.first_ind_dropout_task0)
            self.ind_output_task1 = tf.keras.layers.Dense(1, name = "ind_output_task1")(self.first_ind_dropout_task1)
            self.ind_output_task2 = tf.keras.layers.Dense(1, name = "ind_output_task2")(self.first_ind_dropout_task2)
            self.ind_output_task3 = tf.keras.layers.Dense(1, name = "ind_output_task3")(self.first_ind_dropout_task3)
            self.ind_output_task4 = tf.keras.layers.Dense(1, name = "ind_output_task4")(self.first_ind_dropout_task4)
            self.ind_output_task5 = tf.keras.layers.Dense(1, name = "ind_output_task5")(self.first_ind_dropout_task5)
            self.ind_output_task6 = tf.keras.layers.Dense(1, name = "ind_output_task6")(self.first_ind_dropout_task6)
            self.ind_output_task7 = tf.keras.layers.Dense(1, name = "ind_output_task7")(self.first_ind_dropout_task7)
            self.ind_output_task8 = tf.keras.layers.Dense(1, name = "ind_output_task8")(self.first_ind_dropout_task8)
            self.ind_output_task9 = tf.keras.layers.Dense(1, name = "ind_output_task9")(self.first_ind_dropout_task9)
            #now combine
            self.ind_output_combined = tf.keras.layers.Concatenate(axis = 1)([self.ind_output_task0,
                                                                              self.ind_output_task1,
                                                                              self.ind_output_task2,
                                                                              self.ind_output_task3,
                                                                                  self.ind_output_task4,
                                                                                  self.ind_output_task5,
                                                                                  self.ind_output_task6,
                                                                                  self.ind_output_task7,
                                                                                  self.ind_output_task8,
                                                                                  self.ind_output_task9])
            self.last_shared_output = tf.keras.layers.Dense(self.out_dim, name = "output")(self.ind_output_combined)
            
            self.model_multiple = tf.keras.models.Model(inputs = self.x_input, outputs = self.last_shared_output)
            
    
    def reinitial_layers(self, task_id):
        for i in range(task_id, self.out_dim):
            #reset convolutionals
            exec("self.first_ind_conv_task" + str(i) + " = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', name = 'first_ind_conv_task" + str(i) + "',)(self.second_shared_pool)")
            #reset poolings
            exec("self.first_ind_pool_task" + str(i) + " = tf.keras.layers.MaxPooling2D((2, 2), name = 'first_ind_pool_task"+ str(i) + "')(self.first_ind_conv_task"+ str(i) + ")")
            #reset flattens
            exec("self.first_ind_flatten_task" + str(i) + " = tf.keras.layers.Flatten(name = 'first_ind_flatten_task" + str(i) + "')(self.first_ind_pool_task" + str(i) + ")")
            #reset dense
            exec("self.first_ind_dense_task" + str(i) + " = tf.keras.layers.Dense(1024, activation = 'relu', name = 'first_ind_dense_task" + str(i) + "')(self.first_ind_flatten_task" + str(i) + ")")
            #reset dropout
            exec("self.first_ind_dropout_task" + str(i) + " = tf.keras.layers.Dropout(0.2)(self.first_ind_dense_task" + str(i) + ")")
            #reset output
            exec("self.ind_output_task" + str(i) + " = tf.keras.layers.Dense(1, name = 'ind_output_task" + str(i) + "')(self.first_ind_dense_task" + str(i) + ")")
            
    def save_mean_weights(self, task_id):
        
        model_weights = pd.DataFrame([])
        
        if self.net_mode == "multi-head" and self.train_mode == [0]:
            for i in range(0, len(self.model_multiple.layers)):
                if len(self.model_multiple.layers[i].get_weights()) == 2 and self.model_multiple.layers[i].name != "output": #has weights and bias
                    layer_name = self.model_multiple.layers[i].name
                    mean = tf.reduce_mean(self.model_multiple.layers[i].get_weights()[0]).numpy()
                    median = 0#tfp.stats.percentile(self.model_multiple.layers[i].get_weights()[0], q=50.).numpy()
                    
                    model_weights = pd.concat([model_weights, pd.DataFrame([[task_id, layer_name, mean, median]],
                                                                           columns = ["task_id", "name", "mean_weight", "median_weight"])], axis = 0)

        elif self.net_mode == "multi-head" and (self.train_mode == [100] or self.train_mode == [1000] or self.train_mode == [10000]):
            for i in range(0, len(self.svm_model_multiple.layers)):
                if len(self.svm_model_multiple.layers[i].get_weights()) == 2 and self.svm_model_multiple.layers[i].name != "output": #has weights and bias
                    layer_name = self.svm_model_multiple.layers[i].name
                    mean = tf.reduce_mean(self.svm_model_multiple.layers[i].get_weights()[0]).numpy()
                    median = 0#tfp.stats.percentile(self.svm_model_multiple.layers[i].get_weights()[0], q=50.).numpy()
                    
                    model_weights = pd.concat([model_weights, pd.DataFrame([[task_id, layer_name, mean, median]], 
                                                                     columns = ["task_id", "name", "mean_weight", "median_weight"])], axis = 0)
                    
                        
        elif self.net_mode == "per-task"  and self.train_mode == [0]:
            for i in range(0, len(self.models[task_id].layers)):
                if len(self.models[task_id].layers[i].get_weights()) == 2 and self.models[task_id].layers[i].name != "output": #has weights and bias
                    layer_name = self.models[task_id].layers[i].name
                    mean = tf.reduce_mean(self.models[task_id].layers[i].get_weights()[0]).numpy()
                    median = 0#tfp.stats.percentile(self.models[task_id].layers[i].get_weights()[0], q=50.).numpy()
                    
                    model_weights = pd.concat([model_weights, pd.DataFrame([[task_id, layer_name, mean, median]], 
                                                                     columns = ["task_id", "name", "mean_weight", "median_weight"])], axis = 0)
        
        model_weights.to_csv(os.getcwd() + '\\model_weights\\' + 'method_' + str(self.train_mode[0]) + '_task' + str(task_id) + '_netmode' + self.net_mode + '_modelweights.csv', index = False)
        
    def freeze_layers(self, task_id):
        #freeze for subsequent tasks
        for i in range(task_id + 1, self.out_dim):
            #reset convolutionals
            exec("self.first_ind_conv_task" + str(i) + ".trainable = False")
            #reset poolings
            exec("self.first_ind_pool_task" + str(i) + ".trainable = False")
            #reset flattens
            exec("self.first_ind_flatten_task" + str(i) + ".trainable = False")
            #reset dense
            exec("self.first_ind_dense_task" + str(i) + ".trainable = False")
            #reset dropout
            exec("self.first_ind_dropout_task" + str(i) + ".trainable = False")
            #reset output
            exec("self.ind_output_task" + str(i) + ".trainable = False")

            
    # set vanilla loss    
    def vanilla_loss(self, y_true, y_pred):
        
        if self.net_mode == "single" or self.net_mode == "multi-head":
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
        elif self.net_mode == "per-task":
            return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0))
        
    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
            # computer Fisher information for each parameter

            # initialize Fisher information for most recent task
            self.fisher = []
            for v in range(len(self.var_list)):
                self.fisher.append(np.zeros(self.var_list[v].get_shape().as_list()))

            # sampling a random class from softmax
            probs = tf.nn.softmax(self.y)
            class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

            if(plot_diffs):
                # track differences in mean Fisher info
                prev_fisher = deepcopy(self.fisher)
                mean_diffs = np.zeros(0)

            for i in range(num_samples):
                # select random input image
                im_ind = np.random.randint(imgset.shape[0])
                # compute first-order derivatives
                self.ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
                ders = self.ders
                # square the derivatives and add to total
                for v in range(len(self.fisher)):
                    self.fisher[v] += np.square(ders[v])
                if(plot_diffs):
                    if i % disp_freq == 0 and i > 0:
                        # recording mean diffs of F
                        F_diff = 0
                        for v in range(len(self.fisher)):
                            F_diff += np.sum(np.absolute(self.fisher[v]/(i+1) - prev_fisher[v]))
                        mean_diff = np.mean(F_diff)
                        mean_diffs = np.append(mean_diffs, mean_diff)
                        for v in range(len(self.fisher)):
                            prev_fisher[v] = self.fisher[v]/(i+1)
                        plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                        plt.xlabel("Number of samples")
                        plt.ylabel("Mean absolute Fisher difference")
                        display.display(plt.gcf())
                        display.clear_output(wait=True)

            # divide totals by number of samples
            for v in range(len(self.fisher)):
                self.fisher[v] /= num_samples

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())  # without sess ??

    def restore(self):
        # reassign optimal weights for latest taskhinge?
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                #sess.run(self.var_list[v].assign(self.star_vars[v]))
                self.var_list[v].assign(self.star_vars[v])

    # having doubts with the graph here! I think let's try tensor-board maybe :D
    def set_ewc_loss(self, lam):
        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        # I feel in the third step loss the values of fisher and star_vars would change for the
        # earlier task as well! Does the graph save values of the vars as well ?
        for v in range(len(self.var_list)):
            self.ewc_loss += (lam / 2) * tf.reduce_sum(tf.multiply(self.fisher[v].astype(np.float32),
                                                                   tf.square(self.var_list[v] - self.star_vars[v])))

        self.train_step = tf.train.GradientDescentOptimizer(.1).minimize(self.ewc_loss)
        
        
    #sparse connections in output layer - for making sure there is a one to one connection between output of subnetworks and corresponding unit in the final output layer
    def kill_output_connections(self):
        
        if self.train_mode == [100] or self.train_mode == [1000] or self.train_mode == [10000]:
            for i in range(0, len(self.svm_model_multiple.layers)):
                if self.svm_model_multiple.layers[i].name == "output": 
                    #get weights
                    weights = self.svm_model_multiple.layers[i].get_weights()[0]
                    #get biases 
                    biases = self.svm_model_multiple.layers[i].get_weights()[1]
                    
                    #now, set to zero corresponding connections - just leave a diagonal
                    weights_final = np.zeros((weights.shape))
                    biases_final = np.zeros((biases.shape))
                    
                    for j in range(0, weights.shape[0]):
                        weights_final[j][j] = weights[j][j]
                        biases_final[j] = biases[j]
                    
                    #replace weights
                    self.svm_model_multiple.layers[i].set_weights([weights_final, biases_final])
                    break
        
        else:
            for i in range(0, len(self.model_multiple.layers)):
                if self.model_multiple.layers[i].name == "output": 
                    #get weights
                    weights = self.model_multiple.layers[i].get_weights()[0]
                    #get biases 
                    biases = self.model_multiple.layers[i].get_weights()[1]
                    
                    #now, set to zero corresponding connections - just leave a diagonal
                    weights_final = np.zeros((weights.shape))
                    biases_final = np.zeros((biases.shape))
                    
                    for j in range(0, weights.shape[0]):
                        weights_final[j][j] = weights[j][j]
                        biases_final[j] = biases[j]
                    
                    #replace weights
                    self.model_multiple.layers[i].set_weights([weights_final, biases_final])   
        
    def set_nusvm_loss(self, y_true, y_pred):
        
        #####function to debug ####
        def get_margin_examples(batch_id, y, y_true, out_dim, err, b2):
            #nu-svm loss
            rho_list = [] #one per class
            batch_size = y.shape[0]        
            
            #keep indexes of anchor examples per class, for next iteration
            pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index
            neg_indexes_list = [[]]  #a row is a class for all classes except that one (negative), each column is an instance index
                    
            for i in range(0, out_dim):
                pos_idx = i
                neg_idx = [x for x in range(0, out_dim) if x != pos_idx]
                                
                pos_y = y[:, pos_idx]
                pos_y_true = y_true[:, pos_idx]
                neg_y = y[:, neg_idx]
                neg_y_true = y_true[:, neg_idx]
                
                pos_margin_idx = np.where((pos_y >= 1 - err) & (pos_y <= 1 + err))[0] #np.where((pos_y >= 1 - err) & (pos_y <= 1 + err))[0]
                pos_margin_idx = pos_margin_idx[np.isin(pos_margin_idx, pos_y_true)]
                pos_margin_idx_y = np.repeat(pos_idx, len(pos_margin_idx))
                
                neg_margin_idx, neg_margin_idx_y = np.where((neg_y <= -1 + err) & (neg_y >= -1 - err)) #np.where((neg_y <= -1 + err) & (neg_y >= -1 - err)) #tuple with rows and columns
                neg_margin_idx_keep = np.where(neg_margin_idx[np.isin(neg_margin_idx, neg_y_true)])
                neg_margin_idx = neg_margin_idx[neg_margin_idx_keep]
                neg_margin_idx_y = neg_margin_idx_y[neg_margin_idx_keep]
                
                #make sure to have the same number of examples
                if len(pos_margin_idx) < len(neg_margin_idx) and len(pos_margin_idx) > 0:
                    neg_margin_idx = neg_margin_idx[0:len(pos_margin_idx)]
                    neg_margin_idx_y = neg_margin_idx_y[0:len(pos_margin_idx)]
                    
                elif len(pos_margin_idx) > len(neg_margin_idx) and len(neg_margin_idx) > 0:
                    pos_margin_idx = pos_margin_idx[0:len(neg_margin_idx)]
                    pos_margin_idx_y = pos_margin_idx_y[0:len(neg_margin_idx)]                        

                #now apply nuSVM formula to calculate rho
                
                pos_final = 0.0 
                neg_final = 0.0
                if len(pos_margin_idx) > 0  and len(neg_margin_idx) > 0:
                    for j in range(0, len(pos_margin_idx)):
                        pos_final += y[pos_margin_idx[j], pos_margin_idx_y[j]] - b2[pos_margin_idx_y[j]]
                    for j in range(0, len(neg_margin_idx)):
                        neg_final += y[neg_margin_idx[j], neg_margin_idx_y[j]] - b2[neg_margin_idx_y[j]]                        
                    
                    rho = ((1/2*(len(pos_margin_idx))) *
                                  np.abs(pos_final - neg_final))
                    
                    rho_list.append(rho)
                    
                else:
                    rho_list.append(1.0)
            
            self.batch_id += 1

            return np.asarray(rho_list, dtype = "float32")
        
        rho_pos_neg = tf.numpy_function(func=get_margin_examples, inp=[tf.constant(self.batch_id, dtype = tf.int32),
                                                                    tf.constant(y_pred, dtype = tf.float32),
                                                                    tf.constant(y_true, dtype = tf.float32),
                                                                    tf.constant(self.out_dim, dtype = tf.int32), 
                                                                    tf.constant(self.err, dtype = tf.float32), 
                                                                    tf.constant(self.svm_model.layers[len(self.svm_model.layers) - 1].get_weights()[1], dtype = tf.float32)], 
                                     Tout=tf.float32) 
        

       
        #update values of rho whenever higher than previous iteration
        previous_rho = self.rho
                        
        self.rho = tf.constant(rho_pos_neg, dtype = tf.float32)
        
        #nu-SVM - optionally with oneclass 
        regularization_loss = 0.0
        
        for i in range(0, len(self.svm_model.layers)):
            if self.svm_model.layers[i].name == "output": #readout layer only
                regularization_loss = tf.reduce_mean(tf.square(self.svm_model.layers[i].get_weights()[0])) #regularise weights of the last (linear) layer
                break
        
        hinge_loss = self.hinge_nusvm(self.rho, y_true, y_pred)
        #hinge_loss = self.hinge_nusvm(self.rho[self.task_id], y_true[:,self.task_id], y_pred[:,self.task_id]) #only of the current task

        #predict with previos model
        #self.predictions_current = self.previous_svm_model_multiple.predict_on_batch(self.batch_x)        
        
        nu_parameter =  tf.constant(0.5, dtype = tf.float32) #always 0.5 as the problem is balanced
        #return regularization_loss + self.penalty_parameter * hinge_loss
        return regularization_loss - (nu_parameter * tf.reduce_mean(self.rho)) + (self.penalty_parameter * hinge_loss) - tf.reduce_mean(self.rho[0:(self.task_id + 1)] - self.previous_rho[0:(self.task_id + 1)])

    
    def set_nusvm_multihead_loss(self, y_true, y_pred):
        
        #####function to debug ####
        def get_margin_examples_multihead(batch_id, y, y_true, out_dim, err, rho, b2):
            #nu-svm loss
            rho_list = rho #one per class
            batch_size = y.shape[0]        
            
            #keep indexes of anchor examples per class, for next iteration
            pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index
            neg_indexes_list = [[]]  #a row is a class for all classes except that one (negative), each column is an instance index
                    
            for i in range(0, self.task_id + 1):
                pos_idx = i
                neg_idx = [x for x in range(0, self.out_dim) if x != pos_idx]
                                
                pos_y = y[:, pos_idx]
                pos_y_true = y_true[:, pos_idx]
                neg_y = y[:, neg_idx]
                neg_y_true = y_true[:, neg_idx]
                
                pos_margin_idx = np.where((pos_y >= 1 - self.err) & (pos_y <= 1 + self.err))[0] #np.where((pos_y >= 1 - err) & (pos_y <= 1 + err))[0]
                pos_margin_idx_true = np.where(pos_y_true == 1)[0]
                pos_margin_idx = pos_margin_idx[np.isin(pos_margin_idx, pos_margin_idx_true)]
                pos_margin_idx_y = np.repeat(pos_idx, len(pos_margin_idx))
                
                neg_margin_idx, neg_margin_idx_y = np.where((neg_y <= -1 + err) & (neg_y >= -1 - err)) #np.where((neg_y <= -1 + err) & (neg_y >= -1 - err)) #tuple with rows and columns
                neg_margin_idx_true = np.where(neg_y_true == 1)[0]
                neg_margin_idx_keep = np.where(np.isin(neg_margin_idx, neg_margin_idx_true))[0]
                neg_margin_idx = neg_margin_idx[np.isin(neg_margin_idx, neg_margin_idx_true)]
                neg_margin_idx_y = neg_margin_idx_y[neg_margin_idx_keep]
                
                #make sure to have the same number of examples
                if len(pos_margin_idx) < len(neg_margin_idx) and len(pos_margin_idx) > 0:
                    neg_margin_idx = neg_margin_idx[0:len(pos_margin_idx)]
                    neg_margin_idx_y = neg_margin_idx_y[0:len(pos_margin_idx)]
                    
                elif len(pos_margin_idx) > len(neg_margin_idx) and len(neg_margin_idx) > 0:
                    pos_margin_idx = pos_margin_idx[0:len(neg_margin_idx)]
                    pos_margin_idx_y = pos_margin_idx_y[0:len(neg_margin_idx)]                        

                
                pos_final = 0.0 
                neg_final = 0.0
                if len(pos_margin_idx) > 0  and len(neg_margin_idx) > 0:
                    for j in range(0, len(pos_margin_idx)):
                        pos_final += y[pos_margin_idx[j], pos_margin_idx_y[j]] - b2[pos_margin_idx_y[j]]
                    for j in range(0, len(neg_margin_idx)):
                        neg_final += y[neg_margin_idx[j], neg_margin_idx_y[j]] - b2[neg_margin_idx_y[j]]                       
                    
                    rho_single = 1 + ((1/(2*(len(pos_margin_idx)))) * (pos_final - neg_final))
                    
                    rho_list[i] = rho_single
            
            self.batch_id += 1

            return np.asarray(rho_list, dtype = "float32")
        
        rho_pos_neg = tf.numpy_function(func=get_margin_examples_multihead, inp=[tf.constant(self.batch_id, dtype = tf.int32),
                                                                    tf.constant(y_pred, dtype = tf.float32),
                                                                    tf.constant(y_true, dtype = tf.float32),
                                                                    tf.constant(self.out_dim, dtype = tf.int32), 
                                                                    tf.constant(self.err, dtype = tf.float32),
                                                                    tf.constant(self.rho, dtype = tf.float32),
                                                                    tf.constant(self.svm_model_multiple.layers[len(self.svm_model_multiple.layers) - 1].get_weights()[1], dtype = tf.float32)], 
                                     Tout=tf.float32)    

                       
        new_rho = tf.constant(rho_pos_neg, dtype = tf.float32) #rho for this batch 
       
        #kill unnecesary connections to the output layer - should be one to one from previous layer
        #self.kill_output_connections() 
 
        #nu-SVM - optionally with oneclass 
        regularization_loss = 0.0
        for i in range(0, len(self.svm_model_multiple.layers)):
            #if self.svm_model_multiple.layers[i].name == "output": #readout layer only
            try:
                regularization_loss += tf.reduce_mean(tf.square(self.svm_model_multiple.layers[i].get_weights()[0]) - tf.square(self.previous_svm_model_multiple.layers[i].get_weights()[0]))
            except: 
                #do nothing
                print("")
 #regularise weights of the last (linear) layer
            #break
        
        #nu rho
        nu_rho = (self.nu_parameter * tf.reduce_mean(new_rho[0:(self.task_id+1)]))
        #nu_rho = (self.nu_parameter * tf.reduce_mean(new_rho))
        
        #hinge loss for the current class        
        #hinge_loss = (self.penalty_parameter * tf.reduce_mean(self.hinge_nusvm(new_rho[self.task_id], y_true[:,self.task_id], y_pred[:,self.task_id])))
        #hinge_loss = (self.penalty_parameter * tf.reduce_mean(self.hinge_nusvm(new_rho, y_true, y_pred))) #all tasks
        
        current_examples = np.where(y_true[:,self.task_id] == 1)[0]
        hinge_loss_newtask = (self.penalty_parameter * tf.reduce_mean(self.hinge_nusvm(new_rho, y_true, y_pred))) #all tasks
        #hinge_loss_newtask = (self.penalty_parameter * tf.reduce_mean(self.hinge_nusvm(new_rho, tf.gather(y_true, current_examples), tf.gather(y_pred, current_examples)))) #all tasks
        #hinge_loss_previoustasks = (self.penalty_parameter * tf.reduce_mean(self.hinge_nusvm(new_rho[self.task_id], y_true[:,0:(self.task_id)], y_pred[:,0:(self.task_id)]))) #all tasks
        
        print("HL: " + str(hinge_loss_newtask.numpy()))
        #hinge_loss = (self.penalty_parameter * self.hinge_nusvm(self.rho[self.task_id], y_true[:,self.task_id], y_pred[:,self.task_id])) #only of the current task
        
        #margin size of previous tasks
        margin_size = tf.reduce_mean(new_rho[0:(self.task_id)] - self.previous_rho[0:(self.task_id)])
       
        #predict with current model version (most recent task)
        if self.task_id > 0: #len(y_prev_idx) > 0:
            predictions_current = self.predictions_current
            loss_prev = []
            
            #indices of examples of previous tasks
            #previous_examples = np.where(y_true[:,0:self.task_id] == 1)[0]
            
            for k in range(0, self.out_dim):
                #y_true_k = y_true.numpy()[previous_examples,k]
                #y_true_k = predictions_current[previous_examples,k] #old model
                #y_true_sum = np.sum(y_true, axis = 1)
                #y_pred_k = y_pred.numpy()[previous_examples,k]
                #y_pred_sum = np.sum(y_pred, axis = 1)
                               
                #y_true_k = y_true_k / y_true_sum
                #y_pred_k = y_pred_k / y_pred_sum
                
                #loss_prev += la.norm((predictions_current[k] - tf.reduce_sum(predictions_current[-k])) - (y_pred.numpy()[k] - tf.reduce_sum(y_pred.numpy()[-k])))
                
                #crammer singer
                #y_sum_old = (predictions_current.numpy()[:,[j for j in range(self.out_dim)]] + self.previous_rho.numpy()[[j for j in range(self.out_dim)]]) 
                idx_ex = np.where(predictions_current.numpy()[:,k] == 1)[0]                
                y_sum_old = (predictions_current.numpy()[idx_ex]) + 1
                y_sum_old[idx_ex,k] = y_sum_old[idx_ex,k] - 1 
#                margin_old = tf.reduce_max(y_sum_old, axis = 1) - predictions_current[:,k]
                margin_old = tf.reduce_max(y_sum_old, axis = 1) - y_pred.numpy()[idx_ex,k]
                               
                #y_sum_new = (y_pred.numpy()[:,[j for j in range(self.out_dim) if j != k]] + self.rho.numpy()[[j for j in range(self.out_dim) if j != k]]) 
                #margin_new = tf.reduce_max(y_sum_new, axis = 1) - y_pred[:,k]
                
                #loss_prev.append(tf.reduce_mean(tf.abs(margin_old - margin_new)))

                loss_prev.append(tf.reduce_mean(margin_old))
                
                
            kd = tf.reduce_mean(loss_prev).numpy() #-tf.reduce_mean(loss_prev)
            print("KD: " + str(kd))
            print("Margin Size: " + str(margin_size)) 
            
            #for k in range(0, self.task_id+1):
               # y_prev_idx = np.where(y_true[:,k == 1][0])
                #self.predictions_current_class = self.predictions_current.numpy()[y_prev_idx] #np.repeat(self.rho[k], len(y_prev_idx))
                #self.predictions_new = y_pred.numpy()[y_prev_idx]    
            
                #kd_task = tf.reduce_mean(tf.abs(self.predictions_current - self.predictions_new))
                #kd_task = tf.reduce_mean(tf.square(tf.maximum(tf.abs(self.predictions_current_class - self.predictions_new), 0.)), axis=-1)
                #kd_task = tf.keras.losses.KLDivergence()(self.predictions_current, self.predictions_new).numpy()
                #kd_list.append(kd_task)
                #print("KD Task: " + str(kd_task.numpy()))
            
            #kd = tf.reduce_mean(kd_list)
            #print("HL: " + str(hinge_loss_previoustasks.numpy()))
            #print("MS: " + str(margin_size.numpy()))
#            print("KD: " + str(kd.numpy()))
            
            #self.hinge_loss_perexample = [hinge_loss.numpy()]
            #self.margin_size_perexample = [margin_size.numpy()]
            #self.kd_perexample = [kd.numpy()]
            
            #self.callback.set_values(self.task_id, self.hinge_loss_perexample, self.margin_size_perexample, self.kd_perexample)
            
            #try:
                #self.alpha, self.beta, self.gamma = self.callback.return_values()
            #except:
                #do nothing
                #a = 0
                
            #print(self.hinge_loss_perexample)
            #print(self.margin_size_perexample)
            #print(self.kd_perexample)
            
            #return regularization_loss - nu_rho + self.alpha * hinge_loss
            #return regularization_loss - nu_rho + self.alpha * hinge_loss_newtask + self.beta * (hinge_loss_previoustasks)
            return regularization_loss - nu_rho + self.alpha * hinge_loss_newtask + self.beta * (kd - margin_size)
            #return regularization_loss - nu_rho + self.alpha * hinge_loss_newtask + self.alpha * hinge_loss_previoustasks - self.beta * margin_size + self.gamma * kd
            #return regularization_loss - (nu_parameter * tf.reduce_mean(self.rho)) + (self.penalty_parameter * hinge_loss) - tf.reduce_mean(self.rho[0:(self.task_id)] - previous_rho[0:(self.task_id)]) + 0.5 * tf.abs(tf.reduce_mean(self.predictions_current[:, 0:(self.task_id)] - self.predictions_new[:, 0:(self.task_id)]))

        else:
            #alpha, beta    
            return regularization_loss - nu_rho + self.alpha * hinge_loss_newtask
            #return regularization_loss - (nu_parameter * tf.reduce_mean(self.rho)) + (self.penalty_parameter * hinge_loss) - tf.reduce_mean(self.rho[0:(self.task_id)] - previous_rho[0:(self.task_id)])
        #return regularization_loss + self.penalty_parameter * hinge_loss
        #return regularization_loss - (nu_parameter * tf.reduce_mean(self.rho)) + (self.penalty_parameter * hinge_loss) - tf.reduce_mean(self.rho[0:(self.task_id)] - previous_rho[0:(self.task_id)])
        #return regularization_loss - (nu_parameter * tf.reduce_mean(self.rho)) + hinge_loss + tf.abs(tf.reduce_mean(self.predictions_current[:, 0:(self.task_id)] - y_pred[:, 0:(self.task_id)]))

    def extract_anchors(self, y_pred, y_true):
        #keep indexes of anchor examples per class, for next iteration
        pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index        
        
        #extract SVs and save them for replay in next tasks - classes 0 and 1 only         
        for i in range(0, self.out_dim):
            pos_idx = i
                                
            pos_y = y_pred[:, pos_idx]
            pos_y_true = y_true[:, pos_idx]
                
            pos_margin_idx = np.where((pos_y >= 1 - self.err) & (pos_y <= 1 + self.err))[0] #np.where((pos_y >= 1 - err) & (pos_y <= 1 + err))[0]
            pos_margin_idx_true = np.where(pos_y_true == 1)[0]
            
            #filter true only
            pos_margin_idx = pos_margin_idx[np.isin(pos_margin_idx, pos_margin_idx_true)]
            
            anchors = pos_margin_idx
            
            if len(anchors) > 0:            
                anchors = np.concatenate((np.repeat(i, len(anchors)).reshape(-1,1), 
                                           anchors.reshape(-1,1)), axis = 1)
                
                if len(self.anchors) == 0:
                    self.anchors = anchors
                    self.anchors = self.anchors.astype(int) 
                else:
                    self.anchors = np.append(self.anchors, anchors, axis = 0)
                    self.anchors = self.anchors.astype(int) 
                    
    def extract_anchors_ranking(self, y_pred, y_true):
        #keep indexes of anchor examples per class, for next iteration
        pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index        
        
        #extract SVs and save them for replay in next tasks - classes 0 and 1 only         
        for i in range(0, self.task_id + 1):
            pos_idx = i
                                
            pos_y = y_pred[:, pos_idx]
            pos_y_true = y_true[:, pos_idx]
            
            #filter true only
            #pos_margin_idx_true = np.where(pos_y_true == 1)[0]
            
            #calculate distance to margin, arrange by that distance and select subset
            y_pred_distance_margin = abs(1 - pos_y) #pos_y
            y_pred_distance_margin = y_pred_distance_margin * pos_y_true #y_pred_distance_margin[np.isin(y_pred_distance_margin, pos_margin_idx_true)]
            
            #just keep positives, cause these are correct class + closer
            y_pred_distance_margin[y_pred_distance_margin < 0] = 100000 #only positives (actual class)
            
            y_pred_distance_margin_idx = np.argsort(y_pred_distance_margin) #indexes of sorted array

            #only keep_sv of current class, all the previous keep all (as SVs have been selected already previously!)
            if i == self.task_id:
                y_pred_distance_margin_idx = y_pred_distance_margin_idx[0:int(len(np.where(y_pred_distance_margin != 100000)[0]) * self.keep_sv)]
            else:
                y_pred_distance_margin_idx = np.where(y_pred_distance_margin != 100000)[0]
            
            anchors = y_pred_distance_margin_idx
            
            if len(anchors) > 0:            
                anchors = np.concatenate((np.repeat(i, len(anchors)).reshape(-1,1), 
                                           anchors.reshape(-1,1)), axis = 1)
                
                if len(self.anchors) == 0:
                    self.anchors = anchors
                    self.anchors = self.anchors.astype(int) 
                else:
                    self.anchors = np.append(self.anchors, anchors, axis = 0)
                    self.anchors = self.anchors.astype(int) 
                    
    def extract_anchors_ranking_posneg(self, y_pred, y_true):
        #keep indexes of anchor examples per class, for next iteration
        pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index        
        
        #extract SVs and save them for replay in next tasks - classes 0 and 1 only         
        for i in range(0, self.task_id + 1):
            pos_idx = i
                                
            pos_y = y_pred[:, pos_idx]
            pos_y_true = y_true[:, pos_idx]
            
            neg_y = y_pred[:, [x for x in range(0, self.out_dim) if x != i]]
            neg_y_true = y_true[:, [x for x in range(0, self.out_dim) if x != i]]
            
            #filter true only
            #pos_margin_idx_true = np.where(pos_y_true == 1)[0]
            
            #calculate distance to margin, arrange by that distance and select subset
            y_pred_distance_margin = abs(1 - pos_y) #pos_y
            y_pred_distance_margin = y_pred_distance_margin * pos_y_true #y_pred_distance_margin[np.isin(y_pred_distance_margin, pos_margin_idx_true)]

            y_pred_distance_margin_neg = abs(1 - neg_y) #pos_y
            y_pred_distance_margin_neg = y_pred_distance_margin_neg * neg_y_true #y_pred_distance_margin[np.isin(y_pred_distance_margin, pos_margin_idx_true)]
            
            #just keep positives, cause these are correct class + closer
            y_pred_distance_margin[y_pred_distance_margin < 0] = 100000 #only positives (actual class)
            y_pred_distance_margin_neg[y_pred_distance_margin_neg < 0] = 100000 #only positives (actual class)
            
            y_pred_distance_margin_idx = np.argsort(y_pred_distance_margin) #indexes of sorted array
            y_pred_distance_margin_idx_neg = np.argsort(np.min(y_pred_distance_margin_neg, axis = 1))

            #only keep_sv of current class, all the previous keep all (as SVs have been selected already previously!)
            if i == self.task_id:
                y_pred_distance_margin_idx = y_pred_distance_margin_idx[0:int(len(np.where(y_pred_distance_margin != 100000)[0]) * self.keep_sv)]
                y_pred_distance_margin_idx_neg = y_pred_distance_margin_idx_neg[0:int(len(np.where(y_pred_distance_margin_neg != 100000)[0]) * self.keep_sv)]
            else:
                y_pred_distance_margin_idx = np.where(y_pred_distance_margin != 100000)[0]
                #negatives will still need to be filtered though
                y_pred_distance_margin_idx_neg = y_pred_distance_margin_idx_neg[0:int(len(np.where(y_pred_distance_margin_neg != 100000)[0]) * self.keep_sv)]
            
            anchors = np.concatenate((y_pred_distance_margin_idx, y_pred_distance_margin_idx_neg))
            
            if len(anchors) > 0:            
                anchors = np.concatenate((np.repeat(i, len(anchors)).reshape(-1,1), anchors.reshape(-1,1)), axis = 1)
                
                if len(self.anchors) == 0:
                    self.anchors = anchors
                    self.anchors = self.anchors.astype(int) 
                else:
                    self.anchors = np.append(self.anchors, anchors, axis = 0)
                    self.anchors = self.anchors.astype(int)                     
                    
    def extract_anchors_random(self, y_pred, y_true):
        #keep indexes of anchor examples per class, for next iteration
        pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index        
        
        #extract SVs and save them for replay in next tasks - classes 0 and 1 only         
        for i in range(0, self.task_id + 1):
            pos_idx = i
                                
            pos_y = y_pred[:, pos_idx]
            pos_y_true = y_true[:, pos_idx]
            
            #filter true only
            pos_margin_idx_true = np.where(pos_y_true == 1)[0]
            
            #arrange randomly
            y_pred_distance_margin = pos_y
            y_pred_distance_margin = np.argsort(y_pred_distance_margin)

            #for now it could be less than self.keep_sv
            y_pred_distance_margin = y_pred_distance_margin[np.isin(y_pred_distance_margin, pos_margin_idx_true)]
            np.random.shuffle(y_pred_distance_margin) #indexes of sorted array
            
            #only keep_sv of current class, all the previous keep all (as SVs have been selected already previously!)
            if i == self.task_id:
                y_pred_distance_margin = y_pred_distance_margin[0:int(len(y_pred_distance_margin) * self.keep_sv)]
            
            anchors = y_pred_distance_margin
            
            if len(anchors) > 0:            
                anchors = np.concatenate((np.repeat(i, len(anchors)).reshape(-1,1), 
                                           anchors.reshape(-1,1)), axis = 1)
                
                if len(self.anchors) == 0:
                    self.anchors = anchors
                    self.anchors = self.anchors.astype(int) 
                else:
                    self.anchors = np.append(self.anchors, anchors, axis = 0)
                    self.anchors = self.anchors.astype(int)                    
     
        
    #for two classes only
    def calculate_rho(self, y, y_true, err, b2):
        
        rho_list = self.rho.numpy().tolist()
        for i in range(0, self.task_id + 1):
            pos_idx = i
            neg_idx = [x for x in range(0, self.out_dim) if x != pos_idx]
                                
            pos_y = y[:, pos_idx]
            pos_y_true = y_true[:, pos_idx]
            neg_y = y[:, neg_idx]
            neg_y_true = y_true[:, neg_idx]
                
            pos_margin_idx = np.where((pos_y >= 1 - self.err) & (pos_y <= 1 + self.err))[0] #np.where((pos_y >= 1 - err) & (pos_y <= 1 + err))[0]
            pos_margin_idx_true = np.where(pos_y_true == 1)[0]
            pos_margin_idx = pos_margin_idx[np.isin(pos_margin_idx, pos_margin_idx_true)]
            pos_margin_idx_y = np.repeat(pos_idx, len(pos_margin_idx))
                
            neg_margin_idx = np.where((neg_y <= -1 + self.err) & (neg_y >= -1 - self.err))[0] #np.where((neg_y <= -1 + err) & (neg_y >= -1 - err)) #tuple with rows and columns
            neg_margin_idx_true = np.where(neg_y_true == 1)[0]
            neg_margin_idx = neg_margin_idx[np.isin(neg_margin_idx, neg_margin_idx_true)]
            neg_margin_idx_y = np.repeat(neg_idx, len(neg_margin_idx))
                
            #make sure to have the same number of examples
            if len(pos_margin_idx) < len(neg_margin_idx) and len(pos_margin_idx) > 0:
                neg_margin_idx = neg_margin_idx[0:len(pos_margin_idx)]
                neg_margin_idx_y = neg_margin_idx_y[0:len(pos_margin_idx)]
                        
            elif len(pos_margin_idx) > len(neg_margin_idx) and len(neg_margin_idx) > 0:
                pos_margin_idx = pos_margin_idx[0:len(neg_margin_idx)]
                pos_margin_idx_y = pos_margin_idx_y[0:len(neg_margin_idx)]                        
                    
            pos_final = 0.0 
            neg_final = 0.0
            if len(pos_margin_idx) > 0  and len(neg_margin_idx) > 0:
                for j in range(0, len(pos_margin_idx)):
                    pos_final += y[pos_margin_idx[j], pos_margin_idx_y[j]] - b2[pos_margin_idx_y[j]]
                for j in range(0, len(neg_margin_idx)):
                    neg_final += y[neg_margin_idx[j], neg_margin_idx_y[j]] - b2[neg_margin_idx_y[j]]
                        
                rho_list[i] = 1 + ((1/(2*(len(pos_margin_idx)))) * (pos_final - neg_final))
                    
        self.rho = tf.constant(rho_list, dtype = tf.float32)
        
                    
    def extract_anchors_all(self, y_pred, y_true):
        #keep indexes of anchor examples per class, for next iteration
        pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index
        neg_indexes_list = [[]]  #a row is a class for all classes except that one (negative), each column is an instance index
                    
        for i in range(0, self.out_dim):
            pos_idx = i
            neg_idx = [x for x in range(0, self.out_dim) if x != pos_idx]
                                
            pos_y = y_pred[:, pos_idx]
            pos_y_true = y_true[:, pos_idx]
            neg_y = y_pred[:, neg_idx]
            neg_y_true = y_true[:, neg_idx]
                
            pos_margin_idx = np.where((pos_y >= 1 - self.err) & (pos_y <= 1 + self.err))[0] #np.where((pos_y >= 1 - err) & (pos_y <= 1 + err))[0]
            pos_margin_idx = pos_margin_idx[np.isin(pos_margin_idx, pos_y_true)]
            pos_margin_idx_y = np.repeat(pos_idx, len(pos_margin_idx))
                
            neg_margin_idx, neg_margin_idx_y = np.where((neg_y <= -1 + self.err) & (neg_y >= -1 - self.err)) #np.where((neg_y <= -1 + err) & (neg_y >= -1 - err)) #tuple with rows and columns
            neg_margin_idx_keep = np.where(neg_margin_idx[np.isin(neg_margin_idx, neg_y_true)])
            neg_margin_idx = neg_margin_idx[neg_margin_idx_keep]
            neg_margin_idx_y = neg_margin_idx_y[neg_margin_idx_keep]
                
            #make sure to have the same number of examples
            if len(pos_margin_idx) < len(neg_margin_idx) and len(pos_margin_idx) > 0:
                neg_margin_idx = neg_margin_idx[0:len(pos_margin_idx)]
                neg_margin_idx_y = neg_margin_idx_y[0:len(pos_margin_idx)]
                
            elif len(pos_margin_idx) > len(neg_margin_idx) and len(neg_margin_idx) > 0:
                pos_margin_idx = pos_margin_idx[0:len(neg_margin_idx)]
                pos_margin_idx_y = pos_margin_idx_y[0:len(neg_margin_idx)]
            
            anchors = np.concatenate(pos_margin_idx, neg_margin_idx)
            
            if len(self.anchors) == 0:
                self.anchors = anchors
                self.anchors = self.anchors.astype(int) 
            else:
                self.anchors = np.append(self.anchors, anchors, axis = 0)
                self.anchors = self.anchors.astype(int)   

    def set_nusvm_multihead_loss_withoneclass(self, y_true, y_pred):
        
        #####function to debug ####
        def get_margin_examples_multihead(batch_id, y, y_true, out_dim, err, b2):
            #nu-svm loss
            rho_list = [] #one per class
            batch_size = y.shape[0]        
            
            #keep indexes of anchor examples per class, for next iteration
            pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index
            neg_indexes_list = [[]]  #a row is a class for all classes except that one (negative), each column is an instance index
                    
            for i in range(0, out_dim):
                pos_idx = i
                neg_idx = [x for x in range(0, out_dim) if x != pos_idx]
                                
                pos_y = y[:, pos_idx]
                pos_y_true = y_true[:, pos_idx]
                neg_y = y[:, neg_idx]
                neg_y_true = y_true[:, neg_idx]
                
                pos_margin_idx = np.where((pos_y >= 1 - err) & (pos_y <= 1 + err))[0] #np.where((pos_y >= 1 - err) & (pos_y <= 1 + err))[0]
                pos_margin_idx = pos_margin_idx[np.isin(pos_margin_idx, pos_y_true)]
                pos_margin_idx_y = np.repeat(pos_idx, len(pos_margin_idx))
                
                neg_margin_idx, neg_margin_idx_y = np.where((neg_y <= -1 + err) & (neg_y >= -1 - err)) #np.where((neg_y <= -1 + err) & (neg_y >= -1 - err)) #tuple with rows and columns
                neg_margin_idx_keep = np.where(neg_margin_idx[np.isin(neg_margin_idx, neg_y_true)])
                neg_margin_idx = neg_margin_idx[neg_margin_idx_keep]
                neg_margin_idx_y = neg_margin_idx_y[neg_margin_idx_keep]
                
                #make sure to have the same number of examples
                if len(pos_margin_idx) < len(neg_margin_idx) and len(pos_margin_idx) > 0:
                    neg_margin_idx = neg_margin_idx[0:len(pos_margin_idx)]
                    neg_margin_idx_y = neg_margin_idx_y[0:len(pos_margin_idx)]
                    
                elif len(pos_margin_idx) > len(neg_margin_idx) and len(neg_margin_idx) > 0:
                    pos_margin_idx = pos_margin_idx[0:len(neg_margin_idx)]
                    pos_margin_idx_y = pos_margin_idx_y[0:len(neg_margin_idx)]                        

                #now apply nuSVM formula to calculate rho
                
                pos_final = 0.0 
                neg_final = 0.0
                if len(pos_margin_idx) > 0  and len(neg_margin_idx) > 0:
                    for j in range(0, len(pos_margin_idx)):
                        pos_final += y[pos_margin_idx[j], pos_margin_idx_y[j]] - b2[pos_margin_idx_y[j]]
                    for j in range(0, len(neg_margin_idx)):
                        neg_final += y[neg_margin_idx[j], neg_margin_idx_y[j]] - b2[neg_margin_idx_y[j]]                        
                    
                    rho = ((1/2*(len(pos_margin_idx))) *
                                  np.abs(pos_final - neg_final))
                    
                    rho_list.append(rho)
                    
                else:
                    rho_list.append(1.0)
            
            self.batch_id += 1

            return np.asarray(rho_list, dtype = "float32")
        
        rho_pos_neg = tf.numpy_function(func=get_margin_examples_multihead, inp=[tf.constant(self.batch_id, dtype = tf.int32),
                                                                    tf.constant(y_pred, dtype = tf.float32),
                                                                    tf.constant(y_true, dtype = tf.float32),
                                                                    tf.constant(self.out_dim, dtype = tf.int32), 
                                                                    tf.constant(self.err, dtype = tf.float32), 
                                                                    tf.constant(self.svm_model_multiple.layers[len(self.svm_model_multiple.layers) - 1].get_weights()[1], dtype = tf.float32)], 
                                     Tout=tf.float32)
        

        #update values of rho whenever higher than previous iteration
        previous_rho = self.rho
                        
        self.rho = tf.constant(rho_pos_neg, dtype = tf.float32)
        
        #kill output connections of other tasks - one by one 
        self.kill_output_connections()
        
        #predict with one-class
        self.y_pred_oneclass = np.zeros((self.batch_x.shape[0], self.out_dim))
        
        for k in range(0, self.task_id + 1):
            self.y_pred_oneclass[:,k] = self.list_oneclass_models[k].predict_on_batch(self.batch_x)[:,0]
        
        self.svm_oneclass_model.predict_on_batch(self.batch_x)
        
        #nu-SVM - optionally with oneclass 
        regularization_loss = 0.0
        for i in range(0, len(self.svm_model_multiple.layers)):
            if self.svm_model_multiple.layers[i].name == "output": #readout layer only
                regularization_loss = tf.reduce_mean(tf.square(self.svm_model_multiple.layers[i].get_weights()[0])) #regularise weights of the last (linear) layer
                break
        
        #hinge_loss = self.hinge_nusvm(self.rho, y_true, y_pred + predictions_oneclass_currentbatch)
        hinge_loss = self.hinge_nusvm(self.rho, y_true, y_pred + self.y_pred_oneclass) #only of the current task
                
        nu_parameter =  tf.constant(0.5, dtype = tf.float32) #always 0.5 as the problem is balanced
        #return regularization_loss + self.penalty_parameter * hinge_loss
        return regularization_loss - (nu_parameter * tf.reduce_mean(self.rho)) + (hinge_loss) - tf.reduce_mean(self.rho[0:(self.task_id + 1)] - previous_rho[0:(self.task_id + 1)])

    def set_oneclass_loss(self, y_true, y_pred):
        
        #####function to debug ####
        def get_margin_examples_oneclass(batch_id, y, y_true, out_dim, err, rho):
            #nu-svm loss
            
            rho_list = 0.0 #one per class
            batch_size = y.shape[0]        
            
            #keep indexes of anchor examples per class, for next iteration
            pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index
            neg_indexes_list = [[]]  #a row is a class for all classes except that one (negative), each column is an instance index
                    
            pos_idx = 1
            neg_idx = -1
                                
            pos_y_true = np.where(y_true == 1)[0]
            neg_y_true = np.where(y_true == -1)[0]
                
            pos_margin_idx = pos_y_true
            pos_margin_idx = pos_margin_idx[np.isin(pos_margin_idx, pos_y_true)]
            pos_margin_idx_y = np.repeat(1, len(pos_margin_idx))
                
            neg_margin_idx = neg_y_true #tuple with rows and columns
            neg_margin_idx = neg_margin_idx[np.isin(neg_margin_idx, neg_y_true)]
            neg_margin_idx_y = np.repeat(-1, len(neg_margin_idx))
                         
            #now apply nuSVM formula to calculate rho
                
            points_sum = 0.0
            for j in range(0, len(pos_margin_idx)):
                points_sum += y[pos_margin_idx[j]]
            for j in range(0, len(neg_margin_idx)):
                points_sum += y[neg_margin_idx[j]] 

            #mean
            points_sum = points_sum / (len(pos_margin_idx) + len(neg_margin_idx))
                    
            rho_list = points_sum

            return np.asarray(rho_list, dtype = "float32")
        
        rho_oneclass = tf.numpy_function(func=get_margin_examples_oneclass, inp=[tf.constant(self.batch_id, dtype = tf.int32),
                                                                    tf.constant(y_pred, dtype = tf.float32), 
                                                                    tf.constant(y_true, dtype = tf.float32), 
                                                                    tf.constant(self.out_dim, dtype = tf.int32), 
                                                                    tf.constant(self.err_oneclass, dtype = tf.float32), 
                                                                    tf.constant(self.rho_oneclass, dtype = tf.float32)], 
                                     Tout=tf.float32)
        
                        
        self.rho_oneclass = tf.constant(rho_oneclass, dtype = tf.float32)
        #print('rho one class: ' + str(self.rho_oneclass))
        
        regularization_loss = 0.0
        #one-class SVM
        for i in range(0, len(self.svm_oneclass_model.layers)):
            if self.svm_oneclass_model.layers[i].name == "output": #readout layer only
                regularization_loss = tf.reduce_mean(tf.square(self.svm_oneclass_model.layers[i].get_weights()[0])) #regularise weights of the last (linear) layer
                break        

            
        hinge_loss = self.hinge_nusvm_oneclass(self.rho_oneclass, y_pred)
        #print('hinge loss: ' + str(hinge_loss))
        
        nu_parameter = 0.5
        return (1/2 * regularization_loss) - self.rho_oneclass + ((1/nu_parameter) * hinge_loss)
    
    def train_oneclass(self):
        
        self.svm_oneclass_model.load_weights('./checkpoints/oneclass_model') #load the empty model
        
        self.svm_oneclass_model.compile(tf.keras.optimizers.SGD(learning_rate=0.00001), 
                                                loss = self.set_oneclass_loss, 
                                                metrics = [self.accuracy_svm_oneclass], 
                                                run_eagerly = True)
            
        
        self.svm_oneclass_model.train_on_batch(self.x_input_oneclass, self.y_observed_oneclass)       
        
        model = self.svm_oneclass_model
        
        self.list_oneclass_models = self.list_oneclass_models + [model]
       
        print(self.accuracy_svm_oneclass(self.y_observed_oneclass, self.svm_oneclass_model.predict_on_batch(self.x_input_oneclass)))
        
        
    def set_csvm_loss(self, y_true, y_pred):
        
        #regular C-SVM             
        regularization_loss = 0.0
        for i in range(0, len(self.svm_model.layers)):
            if self.svm_model.layers[i].name == "output": #readout layer only
                regularization_loss = tf.reduce_mean(tf.square(self.svm_model.layers[i].get_weights()[0])) #regularise weights of the last (linear) layer
                break
        
        hinge_loss = tf.reduce_mean(tf.keras.losses.squared_hinge(y_true, y_pred))
                
        return regularization_loss + self.penalty_parameter * hinge_loss
    
    def set_csvm_multihead_loss(self, y_true, y_pred):
        
        #regular C-SVM             
        regularization_loss = 0.0
        self.kill_output_connections()
        
        for i in range(0, len(self.svm_model_multiple.layers)):
            if self.svm_model_multiple.layers[i].name == "output": #readout layer only
                regularization_loss = tf.reduce_mean(tf.square(self.svm_model_multiple.layers[i].get_weights()[0])) #regularise weights of the last (linear) layer
                break
        
        hinge_loss = tf.reduce_mean(tf.keras.losses.squared_hinge(y_true, y_pred))
                
        return regularization_loss + self.penalty_parameter * hinge_loss
      
    def hinge_nusvm(self, rho, y_true, y_pred): #squared hinge
        #binary class (working)
       # return tf.reduce_mean(tf.square(tf.maximum(1 - y_true * y_pred, 0.)), axis=-1)
       return tf.reduce_mean(tf.maximum(self.rho - y_true * y_pred, 0.), axis=-1)
       #k = self.task_id 
       #y_sum = (y_pred.numpy()[:,[j for j in range(self.out_dim) if j != k]] + self.rho.numpy()[[j for j in range(self.out_dim) if j != k]]) 
       #return tf.reduce_mean(tf.reduce_max(y_sum, axis = 1) - y_pred[:,k])
        
        #multiclass hinge
        #y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
        #y_true = math_ops.cast(y_true, y_pred.dtype)
        #pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
        #neg = math_ops.reduce_max((1. - y_true) * y_pred, axis=-1)
        #zero = math_ops.cast(0., y_pred.dtype)
        #return tf.reduce_mean(math_ops.maximum(neg - pos + 1., zero))
        
    
    def hinge_nusvm_perexample(self, rho, y_true, y_pred): #squared hinge
        #return tf.reduce_mean(tf.square(tf.multiply((11/9 - y_pred)*9/2, tf.maximum(rho - y_true * y_pred, 0.))), axis=-1)
        return tf.square(tf.maximum(rho - y_true * y_pred, 0.))
    
    def hinge_nusvm_oneclass(self, rho, y_pred): #squared hinge
        return tf.reduce_mean(tf.square(tf.maximum(rho - y_pred, 0.)))
        #rescaling_factor = 1
        #beta = 1 / (1 - np.exp(-rescaling_factor))
        #return beta * (1 - np.exp(-rescaling_factor * tf.reduce_mean(tf.square(tf.maximum(rho - y_pred, 0.)), axis=-1)))#rescaled 
        
    #svm accuracy metric
    def accuracy_svm(self, y_true, y_pred):
        #closest margin
        #y_pred_max = np.where(y_true * y_pred > 0, y_pred, np.inf)
        #y_pred_min = np.amin(y_pred_max, 1)
        #for i in range(0, y_pred_max.shape[0]):
            #y_pred_max[i] = np.where(y_pred_max[i] == y_pred_min[i], 1, -1) 
        #output = tf.identity(tf.sign(y_pred_max), name = "prediction")

        #max predicted
        #y_pred_max = np.where(y_pred > 0, y_pred, np.inf)
        #y_pred_min = np.amax(y_pred_max, 1)
        #for i in range(0, y_pred_max.shape[0]):
            #y_pred_max[i] = np.where(y_pred_max[i] == y_pred_min[i], 1, -1)
        #output = tf.identity(tf.sign(y_pred_max), name = "prediction")

        
        #max predicted > 1
        #y_pred_max = np.where(y_true * y_pred > 0, y_pred, -1)
        #y_pred_max = np.where(y_pred_max > 0, 1, -1)
        #output = tf.identity(tf.sign(y_pred_max), name = "prediction")
        
        #more restrictive
        #y_pred_max = np.where(y_pred > 0, y_pred, -1)
        #y_pred_min = np.amax(y_pred_max, 1)
        #for i in range(0, y_pred_max.shape[0]):
            #y_pred_max[i] = np.where(y_pred_max[i] == y_pred_min[i], 1, -1)
        #output = tf.identity(tf.sign(y_pred_max), name = "prediction")
        
        #correct_prediction = tf.equal(tf.argmin(tf.maximum(y_pred - 1, 0), 1), tf.argmax(y_true, 1))
        
        #original
        #output = tf.identity(tf.sign(y_pred), name="prediction") #1 or -1 for all (sign of the prediction)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)) #index of the max predicted value, is it the same? - this is only for a binary task
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #convert to float, then take mean
        return accuracy
    
    #one-class svm accuracy metric
    def accuracy_svm_oneclass(self, y_true, y_pred):
        output = tf.identity(tf.sign(y_pred), name="prediction") #1 or -1 for all (sign of the prediction)
        correct_prediction = tf.equal(output, y_true) #are both -1 or 1?
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #convert to float, then take mean
        return accuracy
    
    def accuracy_vanilla(self, y_true, y_pred):
        output = tf.identity(tf.sign(y_pred), name="prediction")
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
    
    def accuracy_vanilla_pertask(self, y_true, y_pred):
        correct_prediction = tf.equal(y_true, tf.round(y_pred))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
        
    def model_train(self, task_id, x, y):
        self.x = tf.reshape(tf.convert_to_tensor(x, dtype=tf.float32), [-1, self.rows_data, self.cols_data, self.channels_data])
        self.x_input = tf.reshape(tf.convert_to_tensor(x, dtype=tf.float32), [-1, self.rows_data, self.cols_data, self.channels_data])
        self.y_observed = tf.convert_to_tensor(y, dtype=tf.float32)
        self.batch_id = 0
        self.task_id = task_id
        
        #checkpointing 
        if task_id > 0 and self.checkpointing == True:
            if self.train_mode == [0]:
                checkpoint_path = os.getcwd() + "\\checkpoints\\cp-{epoch:05d}.ckpt"
                self.model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path)))
            elif self.train_mode == [100] or self.train_mode == [1000]:
                checkpoint_path = os.getcwd() + "\\checkpoints\\cp-{epoch:05d}.ckpt"
                self.svm_model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path)))
                
        #loss, etc
        if (self.train_mode == [0] or self.train_mode == [20]):
            
            if self.net_mode == "single":
                #compile and train with vanilla loss
                    
                self.model.compile(tf.keras.optimizers.SGD(learning_rate=0.001, momentum = 0.9), 
                          loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True), 
                          #loss = self.vanilla_loss,
                          #metrics = [self.accuracy_vanilla],
                          metrics = ["accuracy"], 
                          run_eagerly = True)
            
                self.model.fit(self.x, self.y_observed, batch_size = self.batch_size, verbose = 2, shuffle = True, epochs = 10)
                
            elif self.net_mode == "multi-head":
                #restart layers except of prior tasks
                #if self.reinit_layers:
                    #self.freeze_layers(task_id)
                
                self.model_multiple.compile(tf.keras.optimizers.Adam(learning_rate=0.001), 
                          loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True), 
                          #metrics = [self.accuracy_vanilla],
                          metrics = ["accuracy"], 
                          run_eagerly = True)
            
                self.model_multiple.fit(self.x_input, self.y_observed, batch_size = self.batch_size, verbose = 2, shuffle = True, epochs = 200)
                
                #save summary of model weights
                self.save_mean_weights(task_id)
            
            elif self.net_mode == "per-task":
                
                self.models[task_id].compile(tf.keras.optimizers.SGD(learning_rate=0.01), 
                          loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), 
                          #metrics = [self.accuracy_vanilla_pertask],
                          metrics = ["accuracy"], 
                          run_eagerly = True)
                
                self.models[task_id].fit(self.x_input, self.y_observed, batch_size = self.batch_size, verbose = 2, shuffle = True, epochs = 10)
                
                #save summary of model weights
                self.save_mean_weights(task_id)
                
        elif self.train_mode == [1000]:
            if self.net_mode == "single":
            
                self.anchors = [] #restart anchors for each task
                self.previous_rho = self.rho
                
                self.svm_model.compile(tf.keras.optimizers.SGD(learning_rate=0.001, momentum = 0.9), 
                          loss = self.set_nusvm_loss, 
                          metrics = [self.accuracy_svm], 
                          run_eagerly = True)
            
                self.svm_model.fit(self.x, self.y_observed, batch_size = self.batch_size, verbose = 2, shuffle = True, epochs = 10)
                
                #store anchors
                self.extract_anchors(self.svm_model.predict(self.x), self.y_observed.numpy())
            
            else: 
                print("training multi-head")
                
                #obtain number of examples per task to determine values of alpha and beta
                self.alpha, self.beta = self.obtain_number_examples()
                
                self.anchors = [] #restart anchors for each task
                self.previous_rho = self.rho
                #self.alpha = 0.5 #hinge
                #self.beta = 0.8 #margin size change
                #self.gamma = 0.8 #knowledge distillation
                
                if self.task_id != 0:
                    self.previous_svm_model_multiple.load_weights('./checkpoints/previous_svm_model_multiple')
                    self.predictions_current_all = self.previous_svm_model_multiple.predict_on_batch(self.x_input)
                    
                #load empty model in any task
                #self.svm_model_multiple.load_weights('./checkpoints/svm_model_multiple') #load the empty model
                
                self.svm_model_multiple.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), 
                                            loss = self.set_nusvm_multihead_loss, 
                                            metrics = [self.accuracy_svm], 
                                            run_eagerly = True)
                
                #self.svm_model_multiple.fit(self.x_input, self.y_observed, batch_size = self.batch_size, shuffle = True, epochs = 10)
                self.svm_model_multiple.fit(self.get_batches(self.x_input, self.y_observed, self.predictions_current_all, self.batch_size), epochs = 200, steps_per_epoch = self.x_input.shape[0] // self.batch_size, callbacks = self.callback)
                
                #store anchors
                if self.type_sv == "ranking":
                    self.extract_anchors_ranking(self.svm_model_multiple.predict(self.x_input), self.y_observed.numpy())
                    #print("ANCHORS SIZE: ", str(self.anchors.shape))
                elif self.type_sv == "ranking_posneg":
                    self.extract_anchors_ranking_posneg(self.svm_model_multiple.predict(self.x_input), self.y_observed.numpy())                    
                elif self.type_sv == "random":
                    self.extract_anchors_random(self.svm_model_multiple.predict(self.x_input), self.y_observed.numpy())
                elif self.type_sv == "distance":
                    self.extract_anchors(self.svm_model_multiple.predict(self.x_input), self.y_observed.numpy())
                
                #calculate rho
                self.calculate_rho(self.svm_model_multiple.predict_on_batch(self.x_input), self.y_observed.numpy(), self.err, self.svm_model_multiple.layers[len(self.svm_model_multiple.layers) - 1].get_weights()[1])
                
                #save summary of model weights
                self.save_mean_weights(task_id)
                
                self.previous_svm_model_multiple.set_weights(self.svm_model_multiple.get_weights())
                self.previous_svm_model_multiple.save_weights('./checkpoints/previous_svm_model_multiple')
                
                #predict all and augment the dataset, for each of the classes
                # current_task = self.task_id
                # self.list_oneclass_models = []
                
                # for j in range(0, current_task + 1):
                #     self.task_id = j
                #     self.predict_all()
                
                #     #retrain with one class
                #     self.train_oneclass()
                    
                # self.svm_model_multiple.compile(tf.keras.optimizers.SGD(learning_rate=0.001), 
                #                                 loss = self.set_nusvm_multihead_loss_withoneclass, 
                #                                 metrics = [self.accuracy_svm], 
                #                                 run_eagerly = True)
                                    
                # self.svm_model_multiple.fit(self.get_batches(self.x_input, self.y_observed, self.batch_size), epochs = 10, steps_per_epoch = self.x_input.shape[0] // self.batch_size)
            
        elif self.train_mode == [100]:
            
            if self.net_mode == "single": #c-svm
            
                self.svm_model.compile(tf.keras.optimizers.SGD(learning_rate=0.001, momentum = 0.9), 
                      loss = self.set_csvm_loss, 
                      metrics = [self.accuracy_svm], 
                      run_eagerly = True)
        
                self.svm_model.fit(self.x, self.y_observed, batch_size = self.batch_size, verbose = 2, shuffle = True, epochs = 200)
            
            elif self.net_mode == "multi-head":
                
                print("training multi-head")
                
                #restart layers except of prior tasks
                if self.reinit_layers:
                    self.reinitial_layers(task_id)
                
                self.svm_model_multiple.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), 
                                            loss = self.set_csvm_multihead_loss, 
                                            metrics = [self.accuracy_svm], 
                                            run_eagerly = True)
                
                self.svm_model_multiple.fit(self.x_input, self.y_observed, batch_size = self.batch_size, verbose = 2, shuffle = True, epochs = 200)
                
                #save summary of model weights
                self.save_mean_weights(task_id)
                
        elif self.train_mode == [10000]:
            if self.net_mode == "multi-head":

                self.anchors = [] #restart anchors for each task
                
                #remove checkpoint oneclass file before starting the sequence
                if self.task_id == 0:
                    for f in os.listdir('./checkpoints/'):
                        os.remove(os.path.join('./checkpoints/', f))
                
                #first optimise one-class models - just task 0 so far                
                self.svm_model_multiple.compile(tf.keras.optimizers.SGD(learning_rate=0.001), 
                                            loss = self.set_nusvm_multihead_loss_withoneclass, 
                                            metrics = [self.accuracy_svm], 
                                            run_eagerly = True)
                
                #self.svm_model_multiple.fit(self.x_input, self.y_observed, batch_size = self.batch_size, epochs = 10, steps_per_epoch = self.x_input.shape[0] // self.batch_size)
                self.svm_model_multiple.fit(self.get_batches(self.x_input, self.y_observed, self.batch_size), epochs = 10, steps_per_epoch = self.x_input.shape[0] // self.batch_size)
                
                #store anchors
                self.extract_anchors(self.svm_model_multiple.predict(self.x_input), self.y_observed.numpy())

                #save summary of model weights
                self.save_mean_weights(task_id)
                
        
        if self.net_mode == "multi-head":
            dot_img_file = 'model_multiple.png'
            #tf.keras.utils.plot_model(self.svm_model_multiple, to_file=dot_img_file, show_shapes=True)
        
        elif self.net_mode == "per-task":
            dot_img_file = 'model_per_task' + str(task_id) + '.png'
            tf.keras.utils.plot_model(self.models[task_id], to_file=dot_img_file, show_shapes=True)
            
    def model_predict(self, x, y, task_id, t):
        self.x = tf.reshape(tf.convert_to_tensor(x, dtype=tf.float32), [-1, self.rows_data, self.cols_data, self.channels_data])
        self.y_observed = tf.convert_to_tensor(y, dtype=tf.float32)
        
        
        if self.train_mode == [100]:
            if self.net_mode == "single":
                y_pred_test = self.svm_model.predict(self.x) 
                return self.accuracy_svm(self.y_observed, y_pred_test)
            
            elif self.net_mode == "multi-head":
                y_pred_test = self.svm_model_multiple.predict(self.x) 
                return self.accuracy_svm(self.y_observed, y_pred_test)                
        
        elif self.train_mode == [1000] or self.train_mode == [10000]:
            if self.net_mode == "single":
                y_pred_test = self.svm_model.predict(self.x) 
                return self.accuracy_svm(self.y_observed, y_pred_test)
            elif self.net_mode == "multi-head":
                y_pred_test = self.svm_model_multiple.predict(self.x) 
                pd.DataFrame( self.svm_model_multiple.predict_on_batch(self.x) ).to_csv("predictions_" + str(task_id) + "_ " + str(t) + ".csv")
                return self.accuracy_svm(self.y_observed, y_pred_test)
    
        else:
            if self.net_mode == "single":
                score, accuracy = self.model.evaluate(self.x, self.y_observed)
            elif self.net_mode == "multi-head":
                score, accuracy = self.model_multiple.evaluate(self.x, self.y_observed)
            elif self.net_mode == "per-task":
                score, accuracy = self.models[task_id].evaluate(self.x, self.y_observed)
                #print(self.models[task_id].predict(self.x))
                
            return accuracy
        
        
    #other functions misc
   # batch generator
    def get_batches(self, X, Y, predictions_current_all, batch_size):
        
        while 1:
            n_samples = X.shape[0]
        
            # Shuffle at the start of epoch
            indices = np.arange(n_samples)
            
            np.random.shuffle(indices)
            batch_idx = indices[0:batch_size]
                
            #store for one-class
            self.batch_x = tf.convert_to_tensor(np.array(X)[batch_idx], dtype = tf.float32)
            self.batch_y = tf.convert_to_tensor(np.array(Y)[batch_idx], dtype = tf.float32)
            
            if self.task_id > 0:
                self.predictions_current = tf.convert_to_tensor(np.array(predictions_current_all)[batch_idx], dtype = tf.float32)
        
            yield self.batch_x, self.batch_y
            
    def predict_all(self): 
        #create input for oneclass model of the current class
        x_pos = self.x_input.numpy()[np.where(self.y_observed.numpy()[:,self.task_id] == 1)[0]]
        self.x_input_oneclass = tf.reshape(x_pos, [x_pos.shape[0], self.rows_data, self.cols_data, self.channels_data])
        
        #predict all
        preds = self.svm_model_multiple.predict(self.x_input)
        
        for i in range(0, preds.shape[0]):
            if preds[i, self.task_id] >= 1 - self.err and preds[i, self.task_id] <= 1 + self.err and self.y_observed[i,self.task_id] == -1: #this point is close to the margin of the current positive class but is not labeled in that class
                x = self.x_input[i]
                
                self.x_input_oneclass = tf.concat([self.x_input_oneclass, tf.reshape(x, [1, self.rows_data, self.cols_data, self.channels_data])], 0)
        
        y_pos = np.repeat(1, self.x_input_oneclass.numpy().shape[0])
        self.y_observed_oneclass = tf.convert_to_tensor(y_pos, dtype=tf.float32)

        print(self.x_input_oneclass.numpy().shape[0])
        print(self.y_observed_oneclass.numpy().shape[0])
        
                
    def obtain_number_examples(self):
        number_examples_newtask = 0
        number_examples_oldtasks = len(np.where(self.y_observed.numpy()[:,0] == 1)[0])
        
        for i in range(0, self.task_id + 1):
            if i == self.task_id:
                number_examples_newtask = len(np.where(self.y_observed.numpy()[:,i] == 1)[0])
            else:
                number_examples_oldtasks = min(number_examples_oldtasks, len(np.where(self.y_observed.numpy()[:,i] == 1)[0]))
            
        beta = 1 - (number_examples_oldtasks / self.y_observed.numpy().shape[0])
        alpha = 1 - ((number_examples_newtask - number_examples_oldtasks) / self.y_observed.numpy().shape[0])
        
        print(number_examples_oldtasks)
        print(number_examples_newtask)
        print("Alpha: "+ str(alpha) + " Beta: "+ str(beta))
        return alpha, beta
                
            
            
                    

            
