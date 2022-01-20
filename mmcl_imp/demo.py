# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:59:31 2020

@author: Diana
"""


import tensorflow as tf
from copy import  deepcopy
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
#from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import pandas as pd
import random
import numpy
import sys
from tensorflow.keras.utils import to_categorical
import pickle

import os
os.chdir('/home/dben652/Documents/CL')
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from mmcl import Model

os.chdir('/home/dben652/Documents/CL/experiments')

last_data_x = None
last_data_y = None

def train_model(dataset, model, train_set, test_sets, num_iters=1000, disp_freq=50, lams=[0], task_id = 0, use_anchors = False, mnist_anchors = None, net_mode = "single", type_sv = "ranking", rep = 0):

    #for each method
    for l in range(len(lams)):
        print("TASK " + str(task_id)) 
        
        #model.restore()
        
        # if lams[l] == 0:
        #     model.vanilla_loss()
        # elif lams[l] == 20: #EWC loss
        #     model.set_ewc_loss(lams[l])
        # elif lams[l] == 1000: #nu-svm loss
        #     model.set_nusvm_loss()

        train_accs = []
        test_accs = []
        train_accs = np.zeros(len(test_sets))
        test_accs = np.zeros(len(test_sets))
        
        train_x = train_set["X_train"]
        train_y = train_set["Y_train"]
        
        #use anchors?
        if use_anchors and mnist_anchors is not None:
            train_x = np.concatenate((train_x, mnist_anchors["X_train"]))
            train_y = np.concatenate((train_y, mnist_anchors["Y_train"]))
    
        if lams[l] == 1000 or lams[l] == 100 or lams[l] == 10000:
            train_y[train_y == 0] = -1
            
        #write how many examples this task is being trained with
        number_examples_pertask = pd.DataFrame([])
        for j in range(0, task_id + 1):
            number_examples = len(np.where(train_y[:, j] == 1)[0])
            number_examples_pertask = pd.concat([number_examples_pertask, pd.DataFrame(np.array([[j, number_examples]]), columns = ['task', 'number_examples'])])

        number_examples_pertask.to_csv(dataset + '_method_' + str(lams[l]) + '_task' + str(task_id) + '_' + net_mode + '_' + type_sv + '_numbertrainingexamples_rep' + str(rep) + '.csv', index = False)
        
        #pd.DataFrame([[task_id, train_x.shape[0]]], columns = ["task", "number_training_examples"]).to_csv(dataset + '_method_' + str(lams[l]) + '_task' + str(task_id) + '_' + net_mode + '_numbertrainingexamples.csv', index = False)
        
        #print examples per task
        # for k in range(0, task_id + 1):
        #     print('Task: ' + str(k) + ": " + str(len(train_y[:,task_id][train_y[:,task_id] == 1])))
        
            
        model.model_train(task_id, train_x, train_y)
            
        #print train accuracies        
        for task in range(len(test_sets)):
            x_test_batch = test_sets[task]["X_train"]
            y_test_batch = test_sets[task]["Y_train"]
            
            #for SVM
            if lams[l] == 1000 or lams[l] == 100 or lams[l] == 10000:
                y_test_batch[y_test_batch == 0] = -1
            
            acc = model.model_predict(x_test_batch, y_test_batch, task, task_id)
            train_accs[task] = acc
                   
        #test on tasks learned so far
        for task in range(len(test_sets)):
            #replace negative class to -1 for SVM
            x_test_batch = test_sets[task]["X_test"]
            y_test_batch = test_sets[task]["Y_test"]

            #for SVM
            if lams[l] == 1000 or lams[l] == 100 or lams[l] == 10000:
                y_test_batch[y_test_batch == 0] = -1
                
            acc = model.model_predict(test_sets[task]["X_test"], test_sets[task]["Y_test"], task, task_id)
            test_accs[task] = acc
            
                                
        #write results to files
        pd.DataFrame([test_accs]).to_csv(dataset + '_method_' + str(lams[l]) + '_task' + str(task_id) + '_' + net_mode + '_' + type_sv + '_testperformances_rep' + str(rep) + '.csv', index = False)
        pd.DataFrame([train_accs]).to_csv(dataset + '_method_' + str(lams[l]) + '_task' + str(task_id) + '_' + net_mode + '_' + type_sv + '_trainperformances_rep' + str(rep) + '.csv', index = False)
        
        #return data this was trained on
        return train_x, train_y

#class per class in mnist (positive and negative)
def split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, positive_class, negative_classes, sample_size = 1):
    #select train sample
    current_y = np.where(Y_train == positive_class)[0]
    s_positive = random.choices(population = current_y, k = int(len(current_y) * sample_size))
    
    negative_y = np.where(np.isin(Y_train, negative_classes))[0]
    s_negative = random.choices(population = negative_y, k = int(len(current_y) * sample_size))
    
    X_train_pos = X_train[s_positive]
    Y_train_pos = Y_train_onehot[s_positive]
    
    X_train_neg = X_train[s_negative]
    Y_train_neg = Y_train_onehot[s_negative]
    
    X_train_sample = np.concatenate((X_train_pos, X_train_neg))
    Y_train_sample = np.concatenate((Y_train_pos, Y_train_neg))
    
    #select all test
    current_y_test = list(np.where(Y_test == positive_class)[0])
    
    X_test_all = X_test[current_y_test]
    Y_test_all = Y_test_onehot[current_y_test]
    
    return {'X_train': X_train_sample, 
            'Y_train': Y_train_sample, 
            'X_test': X_test_all, 
            'Y_test': Y_test_all}


def recover_anchors(mnist_list, last_data_x, last_data_y, anchors):
    x = []
    y = []
    
    x = last_data_x[list(anchors[:,1])] #mnist_list[0]["X_train"][list(anchors[:,0])]
    y = last_data_y[list(anchors[:,1])] #mnist_list[0]["Y_train"][list(anchors[:,0])]        

    return {'X_train': x, 
            'Y_train': y}



#read dataset (e.g. mnist)
dataset = "mnist"
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
#(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
#(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar100.load_data()

#reshape inputs
#X_train = X_train.reshape((60000, 28, 28, 1))
X_train = X_train.astype('float32') / 255

#X_test = X_test.reshape((10000, 28, 28, 1))
X_test = X_test.astype('float32') / 255


Y_train_onehot = to_categorical(Y_train)

Y_test_onehot = to_categorical(Y_test)



#sequential, multiple, best setting
num_classes = 10

#generate task orders
num_orders = 10
task_ids = [x for x in range(0, num_classes)]
task_orders = []
task_orders = np.zeros((num_orders, num_classes))

for task in range(0, num_orders):
    if task < len(task_ids):
        order = random.sample([x for x in task_ids if x != task], k=len(task_ids) - 1)
        order = [task] + order
    else:
        order = random.sample(task_ids, k=len(task_ids))
    
    task_orders[task] = order

task_orders = task_orders.astype(int)

#extract samples for each order
sample_size = 0.5
mnist_list_full = list()

for t in range(0, len(task_orders)):
    task_order = task_orders[t]
    
    mnist_list_order = []
    for i in range(0, num_classes):
        dataset_task = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [i], [x for x in np.arange(num_classes) if x != i], sample_size)
        
        #for all classes except last
        #if i != (num_classes - 1):
            #dataset_task = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [i], [i + 1], sample_size)
        #else:
            #dataset_task = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [i], 0, sample_size)
    
        mnist_list_order = mnist_list_order + [dataset_task]
        
    mnist_list_full.append(mnist_list_order)



#old
# sample_size = 0.2
# dataset0 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [0], [1, 2, 3, 4, 5, 6, 7, 8, 9], sample_size)
# dataset1 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [1], [0, 2, 3, 4, 5, 6, 7, 8, 9], sample_size)
# dataset2 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [2], [0, 1, 3, 4, 5, 6, 7, 8, 9], sample_size)
# dataset3 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [3], [0, 1, 2, 4, 5, 6, 7, 8, 9], sample_size)
# dataset4 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [4], [0, 1, 2, 3, 5, 6, 7, 8, 9], sample_size)
# dataset5 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [5], [0, 1, 2, 3, 4, 6, 7, 8, 9], sample_size)
# dataset6 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [6], [0, 1, 2, 3, 4, 5, 7, 8, 9], sample_size)
# dataset7 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [7], [0, 1, 2, 3, 4, 5, 6, 8, 9], sample_size)
# dataset8 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [8], [0, 1, 2, 3, 4, 5, 6, 7, 9], sample_size)
# dataset9 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [9], [0, 1, 2, 3, 4, 5, 6, 7, 8], sample_size)

# mnist_list = [dataset0, dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9]


#separate tasks sequentially (single class)
# sample_size = 0.5
# dataset0 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [0], [1, 2, 3, 4, 5, 6, 7, 8, 9], sample_size)
# dataset1 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [1], [0, 2, 3, 4, 5, 6, 7, 8, 9], sample_size)
# dataset2 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [2], [0, 1, 3, 4, 5, 6, 7, 8, 9], sample_size)
# dataset3 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [3], [0, 1, 2, 4, 5, 6, 7, 8, 9], sample_size)
# dataset4 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [4], [0, 1, 2, 3, 5, 6, 7, 8, 9], sample_size)
# dataset5 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [5], [0, 1, 2, 3, 4, 6, 7, 8, 9], sample_size)
# dataset6 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [6], [0, 1, 2, 3, 4, 5, 7, 8, 9], sample_size)
# dataset7 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [7], [0, 1, 2, 3, 4, 5, 6, 8, 9], sample_size)
# dataset8 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [8], [0, 1, 2, 3, 4, 5, 6, 7, 9], sample_size)
# dataset9 = split_mnist_singleclass(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [9], [0, 1, 2, 3, 4, 5, 6, 7, 8], sample_size)

# mnist_list = [dataset0, dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9]


#one task for all, all data, with sampling
#indexes = random.choices(population = range(0, Y_train_onehot.shape[0]), k = int(Y_train_onehot.shape[0] * sample_size))
#mnist_list = []
#mnist_list_full = list()

#dataset0 = dict({'X_train': X_train[indexes], 
            #'Y_train': Y_train_onehot[indexes], 
            #'X_test': X_test, 
            #'Y_test': Y_test_onehot})

#mnist_list = mnist_list + [dataset0]

#mnist_list_full.append(mnist_list)

#from pickle file
#mnist_list = []
#mnist_list_full = list()

#dataset0 = dict({'X_train': pickle.load(open("mnist_full_anchors_Xtrain.p", "rb")), 
            #'Y_train': pickle.load(open("mnist_full_anchors_Ytrain.p", "rb")), 
            #'X_test': X_test, 
            #'Y_test': Y_test_onehot})

#dataset0["X_train"] = dataset0["X_train"][np.where((dataset0["Y_train"][:, 0] == 1) | (dataset0["Y_train"][:, 1] == 1) | (dataset0["Y_train"][:, 2] == 1))[0],:]
#dataset0["Y_train"] = dataset0["Y_train"][np.where((dataset0["Y_train"][:, 0] == 1) | (dataset0["Y_train"][:, 1] == 1) | (dataset0["Y_train"][:, 2] == 1))[0],:]
#dataset0["X_test"] = dataset0["X_test"][np.where((dataset0["Y_test"][:, 0] == 1) | (dataset0["Y_test"][:, 8] == 1) | (dataset0["Y_test"][:, 2] == 1))[0],:]
#dataset0["Y_test"] = dataset0["Y_test"][np.where((dataset0["Y_test"][:, 0] == 1) | (dataset0["Y_test"][:, 1] == 1) | (dataset0["Y_test"][:, 2] == 1))[0],:]

#mnist_list = mnist_list + [dataset0]
#mnist_list_full.append(mnist_list)

#separate tasks sequentially (positive-negative classes)
# sample_size = 0.2
# dataset0 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [0], [1], sample_size)
# dataset2 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [2], [3], sample_size)
# dataset4 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [4], [5], sample_size)
# dataset6 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [6], [7], sample_size)
# dataset8 = split_mnist(X_train, Y_train, Y_train_onehot, X_test, Y_test, Y_test_onehot, [8], [9], sample_size)

# mnist_list = [dataset0, dataset2, dataset4, dataset6, dataset8]


#for each method under evaluation
mnist_anchors = None
model = None
use_anchors = True
net_mode = "multi-head" #"single" "per-task"  #model with multiple headers or a single model
train_mode = "keep_old_net" #"restart_each_task" #restart the whole network or keep it task after task
reinit_layers = False #reinitialize task-specific layers trained on prior tasks
checkpointing = False
keep_sv = 0.2
type_sv = "random" #"ranking" #"ranking_posneg" "distance" 

#nrows, ncols and nchannels of the current dataset
if dataset == "mnist":
    nrows = mnist_list_full[0][0]["X_train"].shape[1]
    nchannels = 1
    ncols = mnist_list_full[0][0]["X_train"].shape[2] 
elif dataset == "cifar10" or dataset == "cifar100":
    nrows = mnist_list_full[0][0]["X_train"].shape[1]
    ncols = mnist_list_full[0][0]["X_train"].shape[2]
    nchannels = mnist_list_full[0][0]["X_train"].shape[3]

if dataset == "mnist":
    in_dim =  784
    out_dim = 10
elif dataset == "cifar10":
    in_dim =  1024
    out_dim = 10
elif dataset == "cifar100":
    in_dim = 1024
    out_dim = 100
    

#for each repetition
for rep in range(0, 1):
    
    mnist_list = mnist_list_full[rep]
    
    #for different methods
    for method in [[1000]]:
            
        #sess = tf.Session()
                
        # instantiate new model`
        batch_size = 128
        #sess.run()
            
        previous_data = []
            
        model = Model(dataset, in_dim, out_dim, batch_size, method, net_mode, nrows, ncols, nchannels, reinit_layers, checkpointing, keep_sv, type_sv) # simple 2-layer network
        
        #for each task
        for i in range(0, len(mnist_list)):
                
            #restart old network or keep using the same?
            if train_mode == "restart_each_task":
                model = Model(dataset, in_dim, out_dim, batch_size, method, net_mode, nrows, ncols, nchannels, reinit_layers, checkpointing, keep_sv, type_sv)
                
            #get anchors
            if len(model.anchors) > 0 and use_anchors:
                mnist_anchors = recover_anchors(mnist_list, last_data_x, last_data_y, model.anchors) #append anchors task after task
                print("Using anchors of shape: " + str(mnist_anchors["X_train"].shape))
                 
            if i == 0:
                if method == [0] or method == [20]:
                    last_data_x, last_data_y = train_model(dataset, model, mnist_list[i], [mnist_list[i]], 1000, 20, [0], 0, use_anchors, mnist_anchors, net_mode, type_sv, rep) #train model on task-A
            
                elif method == [1000] or method == [100] or method == [10000]: #nu-svm or c-svm
                    last_data_x, last_data_y = train_model(dataset, model, mnist_list[i], [mnist_list[i]], 1000, 20, method, 0, use_anchors, mnist_anchors, net_mode, type_sv, rep) #method 1000 requires more epochs
                    
                previous_data.append(mnist_list[i])
                        
            else:
                # Now compute the fisher and store the star variables, also create mnist 2 
                if method == [20]:
                    #model.compute_fisher(mnist_list[i].validation.images, sess, num_samples=200, plot_diffs=False)
                    model.star() # store the star variables
                    
                previous_data.append(mnist_list[i])
                
                if method == [1000] or method == [100] or method == [10000]:
                    last_data_x, last_data_y = train_model(dataset, model, mnist_list[i], previous_data, 1000, 20, method, i, use_anchors, mnist_anchors, net_mode, type_sv, rep) # train on task-B 
                else:
                    last_data_x, last_data_y = train_model(dataset, model, mnist_list[i], previous_data, 1000, 20, method, i, use_anchors, mnist_anchors, net_mode, type_sv, rep)

        #model = None
            #sess.close()


#save anchors from full model
#import pickle
#mnist_list_anchors_Xtrain = mnist_list[0]["X_train"][list(model.anchors[:,1])]
#mnist_list_anchors_Ytrain = mnist_list[0]["Y_train"][list(model.anchors[:,1])]
#pickle.dump(mnist_list_anchors_Xtrain, open("mnist_full_anchors_Xtrain.p", "wb" ))
#pickle.dump(mnist_list_anchors_Ytrain, open("mnist_full_anchors_Ytrain.p", "wb" ))
 
