
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

from replay import *
from models.model import *
from models.mnist import *
from method import *
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

    no_replay_method = GenericMethod(
        SoftmaxClassificationModel(
            copy_module_list(mnist_network),
            NonFunctionalReplayBuffer()
        ),
        n_epochs = n_epochs
    )

    fixed_replay_method = GenericMethod(
        SoftmaxClassificationModel(
            copy_module_list(mnist_network),
            FixedLengthReplayBuffer()
        ),
        n_epochs = n_epochs
    )

    ce_replay_method = GenericMethod(
        SoftmaxClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer()
        ),
        n_epochs = n_epochs
    )

    svm_replay_method = GenericMethod(
        SVMClassificationModel(
            copy_module_list(mnist_network),
            HeuristicSortedReplayBuffer()
        ),
        n_epochs = n_epochs
    )

    trad_method = GenericMethod(
        SoftmaxClassificationModel(
            copy_module_list(mnist_network),
            NonFunctionalReplayBuffer()
        ),
        n_epochs = n_epochs
    )

    config = Config(debug=True, render=False, device="cuda:0")

    ## 3. Apply created methods to our setting
    no_replay_results = cl_setting.apply(no_replay_method, config=config)
    fixed_replay_results = cl_setting.apply(fixed_replay_method, config=config)
    svm_replay_results = cl_setting.apply(svm_replay_method, config=config)
    ce_replay_results = cl_setting.apply(ce_replay_method, config=config)

    trad_results = trad_setting.apply(trad_method, config=config)


    print_method_results(no_replay_results, "No replay")
    print_method_results(fixed_replay_results, "Fixed length, chronologically ordered replay")
    print_method_results(svm_replay_results, "Fixed length, SVM loss ordered replay")
    print_method_results(ce_replay_results, "Fixed length, CE loss ordered replay")
    print_method_results(trad_results, "Traditional SL")

if __name__ == "__main__":
    main()
