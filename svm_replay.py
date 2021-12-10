
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
from model import *

from collections import deque
from random import sample


class SVMMethod(Method, target_setting=IncrementalSLSetting):
    """ Minimal example of a Method targetting the Class-Incremental CL setting.
    
    For a quick intro to dataclasses, see examples/dataclasses_example.py    
    """

    @dataclass
    class HParams:
        """ Hyper-parameters of the demo model. """

        # Learning rate of the optimizer.
        learning_rate: float = 0.001

    def __init__(self, replay_buffer: ReplayBufferInterface, hparams: HParams = None, svm_head=True):
        self.hparams: SVMMethod.HParams = hparams or self.HParams()
        self.max_epochs: int = 1
        self.early_stop_patience: int = 2

        self.replay_buffer = replay_buffer

        # We will create those when `configure` will be called, before training.
        self.svm_head = svm_head
        self.model: GeneralisedHeadModel
        self.optimizer: torch.optim.Optimizer

    def configure(self, setting: IncrementalSLSetting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """

        if self.svm_head:
            self.model = SVMHeadModel(
                observation_space=setting.observation_space,
                action_space=setting.action_space,
                reward_space=setting.reward_space,
                replay_buffer=self.replay_buffer
            )
        else:
            self.model = SoftmaxHeadModel(
                observation_space=setting.observation_space,
                action_space=setting.action_space,
                reward_space=setting.reward_space,
                replay_buffer=self.replay_buffer
            )

        self.model.to(setting.config.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.learning_rate,
        )

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Example train loop.
        You can do whatever you want with train_env and valid_env here.
        
        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # configure() will have been called by the setting before we get here.
        best_val_loss = inf
        best_epoch = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            print(f"Starting epoch {epoch}")
            postfix = {}
            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.model.shared_step(
                        batch, environment=train_env
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    postfix.update(metrics_dict)
                    train_pbar.set_postfix(postfix)

            # Validation loop:
            self.model.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.0

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.shared_step(
                        batch, environment=valid_env
                    )
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
            if epoch - best_epoch > self.early_stop_patience:
                print(f"Early stopping at epoch {i}.")
                break

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        with torch.no_grad():
            logits = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)


def demo_simple():
    """ Simple demo: Creating and applying a Method onto a Setting. """

    ## 1. Creating the setting:
    setting = ClassIncrementalSetting(dataset="fashionmnist", batch_size=32, nb_tasks=10)

    ## 2. Creating the Methods
    no_replay_method = SVMMethod(NonFunctionalReplayBuffer())
    fixed_replay_method = SVMMethod(FixedLengthReplayBuffer())
    svm_replay_method = SVMMethod(HeuristicSortedReplayBuffer())
    ce_replay_method = SVMMethod(HeuristicSortedReplayBuffer(), svm_head=False)
    config = Config(debug=True, render=False, device="cuda:0")

    ## 3. Apply created methods to our setting
    no_replay_results = setting.apply(no_replay_method, config=config)
    fixed_replay_results = setting.apply(fixed_replay_method, config=config)
    svm_replay_results = setting.apply(svm_replay_method, config=config)
    ce_replay_results = setting.apply(ce_replay_method, config=config)


    print("\nNo replay results (final):")
    for i, metric in enumerate(no_replay_results.final_performance_metrics):
        print(f' - Task {i} accuracy across {metric.n_samples} samples: {metric.accuracy}')
    print(f' - Average accuracy: {np.mean([metric.accuracy for metric in no_replay_results.final_performance_metrics])}')

    print("\nFixed length, naive replay results (final):")
    for i, metric in enumerate(fixed_replay_results.final_performance_metrics):
        print(f' - Task {i} accuracy across {metric.n_samples} samples: {metric.accuracy}')
    print(f' - Average accuracy: {np.mean([metric.accuracy for metric in fixed_replay_results.final_performance_metrics])}')

    print("\nFixed length, SVM loss ordered replay results (final):")
    for i, metric in enumerate(svm_replay_results.final_performance_metrics):
        print(f' - Task {i} accuracy across {metric.n_samples} samples: {metric.accuracy}')
    print(f' - Average accuracy: {np.mean([metric.accuracy for metric in svm_replay_results.final_performance_metrics])}')

    print("\nFixed length, CE loss ordered replay results (final):")
    for i, metric in enumerate(ce_replay_results.final_performance_metrics):
        print(f' - Task {i} accuracy across {metric.n_samples} samples: {metric.accuracy}')
    print(f' - Average accuracy: {np.mean([metric.accuracy for metric in ce_replay_results.final_performance_metrics])}')


if __name__ == "__main__":
    demo_simple()
