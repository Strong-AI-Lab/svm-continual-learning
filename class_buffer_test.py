
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Optional, Union

import gym
import pandas as pd
import tqdm
import torch
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sequoia import Method, Setting
from sequoia.common import Config
from sequoia.settings import Environment
from sequoia.settings.base.results import Results
from sequoia.settings.sl import DomainIncrementalSLSetting, IncrementalSLSetting, ClassIncrementalSetting
from sequoia.settings.sl.incremental.objects import (
    Actions,
    Observations,
    Rewards,
)

from svm.replay.buffer.common import ClassSeparatedHeuristicReplayBuffer, DelayedClassSeparatedHeuristicReplayBuffer, HeuristicSortedReplayBuffer, NonFunctionalReplayBuffer
from svm.replay.tracker.common import ClassDistributionTracker
from svm.replay.heuristic.base import Heuristic
from svm.replay.heuristic.common import InversionHeuristic, LossHeuristic, ProductHeuristic, WeightedSummationHeuristic, SquaringHeuristic
from svm.replay.heuristic.misc import SVMBoundaryHeuristic, ClassRepresentationHeuristic
from svm.models.model import SoftmaxClassificationModel, SVMClassificationModel
from svm.models.mnist import BasicMNISTNetwork
from svm.method.method import GenericMethod
from svm.method.suite import BasicMethodTestingSuite, GridSearchTestingSuite
from svm.util import copy_module_list, print_method_results

from collections import deque
from random import sample

def main():

    ## 1. Creating the settings:
    n_epochs = 1
    n_runs = 3
    n_tasks = 10

    cl_setting = ClassIncrementalSetting(
        dataset="fashionmnist", 
        batch_size=32, 
        nb_tasks=n_tasks,
        initial_increment=10,
        stationary_context=True
    )
    trad_setting = ClassIncrementalSetting(dataset="fashionmnist", batch_size=32, nb_tasks=1)  # setting that mimics iid SL

    ## 2. Creating the Methods
    mnist_network = BasicMNISTNetwork(cl_setting.observation_space['x'].shape[0], cl_setting.action_space.n)
    config = Config(debug=False, render=False, device="cuda:0")
    
    class_prop = 0.5
    def method_creation_fn():

        methods = [
            GenericMethod(
                SVMClassificationModel(
                    copy_module_list(mnist_network),
                    ClassSeparatedHeuristicReplayBuffer(
                        LossHeuristic(should_update=True), reverse_sort=True
                    )
                ),
                f'Loss ordered replay buffer: hard class balancing',
                n_epochs = n_epochs,
                curr_class_prop=class_prop
            ),

            GenericMethod(
                SVMClassificationModel(
                    copy_module_list(mnist_network),
                    ClassSeparatedHeuristicReplayBuffer(
                        SVMBoundaryHeuristic(should_update=True), reverse_sort=True
                    )
                ),
                f'SVM boundary proximity ordered replay buffer: hard class balancing',
                n_epochs = n_epochs,
                curr_class_prop=class_prop
            ),

            GenericMethod(
                SVMClassificationModel(
                    copy_module_list(mnist_network),
                    DelayedClassSeparatedHeuristicReplayBuffer(
                        LossHeuristic(should_update=True), reverse_sort=True
                    )
                ),
                f'Loss ordered replay buffer: hard class balancing, delayed filtering',
                n_epochs = n_epochs,
                curr_class_prop=class_prop
            ),

            GenericMethod(
                SVMClassificationModel(
                    copy_module_list(mnist_network),
                    DelayedClassSeparatedHeuristicReplayBuffer(
                        SVMBoundaryHeuristic(should_update=True), reverse_sort=True
                    )
                ),
                f'SVM boundary proximity ordered replay buffer: hard class balancing, delayed filtering',
                n_epochs = n_epochs,
                curr_class_prop=class_prop
            ),

            GenericMethod(
                SVMClassificationModel(
                    copy_module_list(mnist_network),
                    ClassSeparatedHeuristicReplayBuffer(
                        Heuristic(),
                    )
                ),
                f'Unordered replay buffer: hard class balancing',
                n_epochs = n_epochs,
                curr_class_prop=class_prop
            )
        ]
        
        for method in methods:
            method.model.replay_buffer.add_tracker(ClassDistributionTracker(cl_setting.action_space.n))

        return methods
    
    testing_suite = BasicMethodTestingSuite(method_creation_fn, cl_setting, config, n_runs)
    test_objs = testing_suite.run()

    # don't really need to repeat traditional method experiments as it has less variance in terms of performance, and only serves as a rough upper bound
    trad_method = GenericMethod(
        SoftmaxClassificationModel(
            copy_module_list(mnist_network),
            NonFunctionalReplayBuffer()
        ),
        "Traditional SL method",
        n_epochs = n_epochs
    )
    trad_results = trad_setting.apply(trad_method, config=config)

    # display metrics
    # print(f'Displaying aggregated results across {n_runs} repeat tests...')
    # for test_obj in test_objs:
        
    #     method = test_obj['method_instances'][0]
    #     print(f'\n"{method.description}" results (final):')
    #     for task_i in range(n_tasks):
    #         n_samples = test_obj['raw_results'][0].final_performance_metrics[task_i].n_samples
    #         task_mean_acc = round(test_obj['task_mean_acc'][task_i], 3)
    #         task_acc_std = round(test_obj['task_acc_std'][task_i], 3)
    #         print(f' - Task {task_i} accuracy metrics across {n_samples} samples: mean={task_mean_acc}, std_dev={task_acc_std}')

    #     print(f' - Average accuracy metrics across all tasks: mean={round(test_obj["mean_acc"], 3)}, std_dev={round(test_obj["acc_std"], 3)}')

    #     # also show class distribution
    #     # TODO: show distribution for each of the n runs?
    #     class_dist_tracker = method.model.replay_buffer.get_tracker(ClassDistributionTracker)
    #     print(f' - Class distribution: {class_dist_tracker.get_distribution()}')

    print_method_results(trad_results, trad_method)

    # TODO: 
    #  1. investigate training on neg. examples from other tasks in current task (e.g. 50% pos., 50% neg. (from other task classes))
    #  2. try implementing iCARL or similar, simple CL method to see if this improves on basic class balancing
    #  3. dont sample replay examples from current task / class(es)

    for i, test_obj in enumerate(test_objs):
        method: GenericMethod = test_obj['method_instances'][0]
        mean_matrix = test_obj['mean_obj_matrix']

        print('\n"' + method.description + '" results:')
        print(f' - Final average accuracy: {round(np.mean(mean_matrix[-1,:]), 3)}')

        fig, ax = plt.subplots()
        
        cdict = {
            'red': [(0, 0, 0), (1, 0, 0)],
            'green': [(0, 0.5, 0.5), (1, 1, 1)],
            'blue': [(0, 0, 0), (1, 0, 0)],
            'alpha': [(0, 0, 0), (1, 0.3, 0.3)]
        }
        green_cmap = LinearSegmentedColormap('green_cmap', segmentdata=cdict, N=256)
        ax.matshow(mean_matrix, cmap=green_cmap)

        for (i, j), z in np.ndenumerate(mean_matrix):
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

        plt.xticks(range(0, cl_setting.nb_tasks))
        plt.yticks(range(0, cl_setting.nb_tasks))

        plt.title(method.description)
        plt.show()


if __name__ == "__main__":
    main()
