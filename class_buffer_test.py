
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Optional, Union

import gym
import pandas as pd
import tqdm
import torch
import numpy as np
from numpy import inf

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

from svm.replay.buffer.common import ClassSeparatedHeuristicReplayBuffer, HeuristicSortedReplayBuffer, NonFunctionalReplayBuffer
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
    n_runs = 1
    n_tasks = 10

    cl_setting = ClassIncrementalSetting(dataset="fashionmnist", batch_size=32, nb_tasks=n_tasks)
    trad_setting = ClassIncrementalSetting(dataset="fashionmnist", batch_size=32, nb_tasks=1)  #  setting that mimics iid SL

    ## 2. Creating the Methods
    mnist_network = BasicMNISTNetwork(cl_setting.observation_space['x'].shape[0], cl_setting.action_space.n)
    config = Config(debug=False, render=False, device="cuda:0")
    
    def method_creation_fn():

        # m1 = GenericMethod(
        #     SVMClassificationModel(
        #         copy_module_list(mnist_network),
        #         HeuristicSortedReplayBuffer(
        #             InversionHeuristic(SVMBoundaryHeuristic(), should_update=True), reverse_sort=True
        #         )  
        #     ),
        #     f'SVM boundary proximity ordered replay buffer: no class balancing, no update restriction',
        #     n_epochs = n_epochs
        # )

        # m2 = GenericMethod(
        #     SVMClassificationModel(
        #         copy_module_list(mnist_network),
        #         HeuristicSortedReplayBuffer(
        #             InversionHeuristic(SVMBoundaryHeuristic(), should_update=True), reverse_sort=True
        #         )  
        #     ),
        #     f'SVM boundary proximity ordered replay buffer: no class balancing, with update restriction',
        #     n_epochs = n_epochs
        # )

        # m3 = GenericMethod(
        #     SVMClassificationModel(
        #         copy_module_list(mnist_network),
        #         HeuristicSortedReplayBuffer(WeightedSummationHeuristic(
        #             [InversionHeuristic(SVMBoundaryHeuristic(), should_update=True), ClassRepresentationHeuristic()], 
        #             coeffs=[0.5, 0.5]),  # balance class eq. and svm heuristics
        #             reverse_sort=True
        #         )  
        #     ),
        #     f'SVM boundary proximity ordered replay buffer: soft class balancing, no update restriction',
        #     n_epochs = n_epochs
        # )

        # m4 = GenericMethod(
        #     SVMClassificationModel(
        #         copy_module_list(mnist_network),
        #         HeuristicSortedReplayBuffer(WeightedSummationHeuristic(
        #             [InversionHeuristic(SVMBoundaryHeuristic(), should_update=False), ClassRepresentationHeuristic()], 
        #             coeffs=[0.5, 0.5]),  # balance class eq. and svm heuristics
        #             reverse_sort=True
        #         )  
        #     ),
        #     f'SVM boundary proximity ordered replay buffer: soft class balancing, with update restriction',
        #     n_epochs = n_epochs
        # )

        methods = [
            GenericMethod(
                SVMClassificationModel(
                    copy_module_list(mnist_network),
                    ClassSeparatedHeuristicReplayBuffer(
                        InversionHeuristic(SVMBoundaryHeuristic(), should_update=True), reverse_sort=True
                    )
                ),
                f'SVM boundary proximity ordered replay buffer: hard class balancing, no update restriction',
                n_epochs = n_epochs
            ),

            GenericMethod(
                SVMClassificationModel(
                    copy_module_list(mnist_network),
                    ClassSeparatedHeuristicReplayBuffer(
                        InversionHeuristic(SVMBoundaryHeuristic(), should_update=False), reverse_sort=True
                    )
                ),
                f'SVM boundary proximity ordered replay buffer: hard class balancing, with update restriction',
                n_epochs = n_epochs
            ),

            GenericMethod(
                SVMClassificationModel(
                    copy_module_list(mnist_network),
                    ClassSeparatedHeuristicReplayBuffer(
                        Heuristic(),
                    )
                ),
                f'Unordered replay buffer: hard class balancing',
                n_epochs = n_epochs
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
    print(f'Displaying aggregated results across {n_runs} repeat tests...')
    for test_obj in test_objs:

        method = test_obj['method_instances'][0]
        print(f'\n"{method.description}" results (final):')
        for task_i in range(n_tasks):
            n_samples = test_obj['raw_results'][0].final_performance_metrics[task_i].n_samples
            task_mean_acc = round(test_obj['task_mean_acc'][task_i], 3)
            task_acc_std = round(test_obj['task_acc_std'][task_i], 3)
            print(f' - Task {task_i} accuracy metrics across {n_samples} samples: mean={task_mean_acc}, std_dev={task_acc_std}')

        print(f' - Average accuracy metrics across all tasks: mean={round(test_obj["mean_acc"], 3)}, std_dev={round(test_obj["acc_std"], 3)}')

        # also show class distribution
        # TODO: show distribution for each of the n runs?
        class_dist_tracker = method.model.replay_buffer.get_tracker(ClassDistributionTracker)
        print(f' - Class distribution: {class_dist_tracker.get_distribution()}')

    print_method_results(trad_results, trad_method)

if __name__ == "__main__":
    main()
