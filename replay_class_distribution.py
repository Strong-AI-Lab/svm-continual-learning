
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
from sequoia.settings.sl import DomainIncrementalSLSetting, IncrementalSLSetting, ClassIncrementalSetting
from sequoia.settings.sl.incremental.objects import (
    Actions,
    Observations,
    Rewards,
)

from svm.replay.buffer.common import HeuristicSortedReplayBuffer, NonFunctionalReplayBuffer
from svm.replay.tracker.common import ClassDistributionTracker
from svm.replay.heuristic.common import InversionHeuristic, LossHeuristic, ProductHeuristic, WeightedSummationHeuristic, SquaringHeuristic
from svm.replay.heuristic.misc import SVMBoundaryHeuristic, ClassRepresentationHeuristic
from svm.models.model import SoftmaxClassificationModel, SVMClassificationModel
from svm.models.mnist import BasicMNISTNetwork
from svm.method.method import GenericMethod
from svm.util import copy_module_list, print_method_results

from collections import deque
from random import sample

def main():

    ## 1. Creating the settings:
    cl_setting = ClassIncrementalSetting(dataset="fashionmnist", batch_size=32, nb_tasks=5)
    trad_setting = ClassIncrementalSetting(dataset="fashionmnist", batch_size=32, nb_tasks=1)  #  setting that mimics iid SL

    ## 2. Creating the Methods

    mnist_network = BasicMNISTNetwork(cl_setting.observation_space['x'].shape[0], cl_setting.action_space.n)
    n_epochs = 1

    ce_loss_method = GenericMethod(
        SoftmaxClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer(LossHeuristic())
        ),
        "Fixed length, CE loss ordered replay method",
        n_epochs = n_epochs
    )

    svm_loss_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer(LossHeuristic())
        ),
        "Fixed length, SVM loss ordered replay method",
        n_epochs = n_epochs
    )

    svm_boundary_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer(InversionHeuristic(SVMBoundaryHeuristic()))  # smaller boundary distances are prioritised
        ),
        "Fixed length, SVM boundary proximity ordered replay method",
        n_epochs = n_epochs
    )

    svm_boundary_class_eq_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer(ProductHeuristic([InversionHeuristic(SVMBoundaryHeuristic()), ClassRepresentationHeuristic()]))  # add class equalising heuristic
        ),
        "Fixed length, SVM boundary proximity ordered replay (with class equalisation) method",
        n_epochs = n_epochs
    )

    svm_boundary_class_eq_sqrd_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer(ProductHeuristic([InversionHeuristic(SVMBoundaryHeuristic()), SquaringHeuristic(ClassRepresentationHeuristic())]))  # add class equalising heuristic
        ),
        "Fixed length, SVM boundary proximity ordered replay (with class equalisation, squared) method",
        n_epochs = n_epochs
    )

    reverse_svm_boundary_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer(SVMBoundaryHeuristic())  # larger boundary distances are prioritised
        ),
        "Fixed length, SVM boundary proximity ordered (reverse) replay method",
        n_epochs = n_epochs
    )

    # combines svm loss and boundary proximity metrics
    hybrid_svm_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer(WeightedSummationHeuristic([InversionHeuristic(SVMBoundaryHeuristic()), LossHeuristic()]))
        ),
        "Fixed length, SVM hybrid (inverse boundary proxim., & loss) ordered replay method",
        n_epochs = n_epochs
    )

    reverse_hybrid_svm_method = GenericMethod(  
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer(WeightedSummationHeuristic([InversionHeuristic(SVMBoundaryHeuristic()), LossHeuristic()]))
        ),
        "Fixed length, SVM hybrid (inverse boundary proxim., & loss) ordered (reverse) replay method",
        n_epochs = n_epochs
    )

    trad_method = GenericMethod(
        SoftmaxClassificationModel(
            copy_module_list(mnist_network),
            NonFunctionalReplayBuffer()
        ),
        "Traditional SL method",
        n_epochs = n_epochs
    )

    cl_methods = [
        ce_loss_method,
        svm_loss_method,
        svm_boundary_method,
        svm_boundary_class_eq_method,
        svm_boundary_class_eq_sqrd_method,
        reverse_svm_boundary_method,
        hybrid_svm_method,
        reverse_hybrid_svm_method
    ]
    config = Config(debug=False, render=False, device="cuda:0")

    # apply each method and store results
    results_arr = []
    for method in cl_methods:
        class_dist_tracker = ClassDistributionTracker(cl_setting.action_space.n)
        method.model.replay_buffer.add_tracker(class_dist_tracker)

        results = cl_setting.apply(method, config=config)
        results_arr.append(results)

    trad_results = trad_setting.apply(trad_method, config=config)


    # print results
    for i, results in enumerate(results_arr):
        method = cl_methods[i]
        print_method_results(results, method)

        # also show class distribution
        class_dist_tracker = method.model.replay_buffer.get_tracker(ClassDistributionTracker)
        print(class_dist_tracker.get_distribution())

    print_method_results(trad_results, trad_method)

if __name__ == "__main__":
    main()
