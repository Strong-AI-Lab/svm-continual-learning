
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Optional, Union

import gym
import pandas as pd
import tqdm
import torch
import numpy as np
from numpy import inf
from simple_parsing import ArgumentParser

from sequoia import Method, Setting
from sequoia.common import Config
from sequoia.settings import Environment
from sequoia.settings.sl import DomainIncrementalSLSetting, IncrementalSLSetting, ClassIncrementalSetting
from sequoia.settings.sl.incremental.objects import (
    Actions,
    Observations,
    Rewards,
)
from sequoia.settings.sl.incremental.results import Results
from sequoia.settings.sl.environment import PassiveEnvironment

from replay.buffer import HeuristicSortedReplayBuffer, NonFunctionalReplayBuffer
from replay.heuristic import LossHeuristic, SVMBoundaryHeuristic
from models.model import SoftmaxClassificationModel, SVMClassificationModel
from models.mnist import BasicMNISTNetwork
from method import GenericMethod
from util import copy_module_list, print_method_results

from collections import deque
from random import sample

def main():

    ## 1. Creating the settings:
    cl_setting = ClassIncrementalSetting(dataset="fashionmnist", batch_size=32, nb_tasks=5)
    trad_setting = ClassIncrementalSetting(dataset="fashionmnist", batch_size=32, nb_tasks=1)  #  setting that mimics iid SL

    ## 2. Creating the Methods

    mnist_network = BasicMNISTNetwork(cl_setting.observation_space['x'].shape[0], cl_setting.action_space.n)
    n_epochs = 1

    # no_replay_method = GenericMethod(
    #     SoftmaxClassificationModel(
    #         copy_module_list(mnist_network),
    #         NonFunctionalReplayBuffer()
    #     ),
    #     "No replay method",
    #     n_epochs = n_epochs
    # )

    # fixed_replay_method = GenericMethod(
    #     SoftmaxClassificationModel(
    #         copy_module_list(mnist_network),
    #         FixedLengthReplayBuffer()
    #     ),
    #     "Fixed length, chronologically ordered replay method",
    #     n_epochs = n_epochs
    # )

    ce_loss_method = GenericMethod(
        SoftmaxClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer([LossHeuristic])
        ),
        "Fixed length, CE loss ordered replay method",
        n_epochs = n_epochs
    )

    svm_loss_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer([LossHeuristic])
        ),
        "Fixed length, SVM loss ordered replay method",
        n_epochs = n_epochs
    )

    svm_boundary_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer([SVMBoundaryHeuristic], reverse_sort=True)  # smaller boundary distances are prioritised
        ),
        "Fixed length, SVM boundary proximity ordered (reverse) replay method",
        n_epochs = n_epochs
    )

    reverse_svm_boundary_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer([SVMBoundaryHeuristic], reverse_sort=False)  # larger boundary distances are prioritised
        ),
        "Fixed length, SVM boundary proximity ordered replay method",
        n_epochs = n_epochs
    )

    # combines svm loss and boundary proximity metrics
    hybrid_svm_method = GenericMethod(  
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer([SVMBoundaryHeuristic, LossHeuristic], reverse_sort=False)
        ),
        "Fixed length, SVM boundary proximity ordered replay method",
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

    config = Config(debug=True, render=False, device="cuda:0")

    ## 3. Apply created methods to our setting
    # no_replay_results = cl_setting.apply(no_replay_method, config=config)
    # fixed_replay_results = cl_setting.apply(fixed_replay_method, config=config)
    ce_loss_results = cl_setting.apply(ce_loss_method, config=config)
    svm_loss_results = cl_setting.apply(svm_loss_method, config=config)
    svm_boundary_results = cl_setting.apply(svm_boundary_method, config=config)
    reverse_svm_boundary_results = cl_setting.apply(reverse_svm_boundary_method, config=config)
    trad_results = trad_setting.apply(trad_method, config=config)

    print_method_results(ce_loss_results, ce_loss_method)
    print_method_results(svm_loss_results, svm_loss_method)
    print_method_results(svm_boundary_results, svm_boundary_method)
    print_method_results(reverse_svm_boundary_results, reverse_svm_boundary_method)
    print_method_results(trad_results, trad_method)

if __name__ == "__main__":
    main()
