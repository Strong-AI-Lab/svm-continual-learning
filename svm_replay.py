""" Demo: Creates a simple new method and applies it to a single CL setting.
"""
from argparse import Namespace
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Optional, Union

from collections import defaultdict
from pathlib import Path

import gym
import pandas as pd
import tqdm
import torch
import numpy as np
from numpy import inf
from gym import spaces
from torch import Tensor, nn
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

from collections import deque
from random import sample

class ReplayExample():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class HeuristicReplayExample(ReplayExample):

    # Replay example that has an associated heuristic, to be used for preferential replay mechanisms

    def __init__(self, x, y, h_value):
        super().__init__(x, y)
        self.h_value = h_value

class ReplayBufferInterface():
    
    def add_examples(self, examples: List[ReplayExample]):
        raise NotImplementedError

    def get_examples(self, num_examples: int, random: bool = True):
        raise NotImplementedError

class NonFunctionalReplayBuffer(ReplayBufferInterface):

    # "Implementation" of replay buffer that does not do anything

    def add_examples(self, examples: List[ReplayExample]):
        pass

    def get_examples(self, num_examples: int, random: bool = True):
        return []

class FixedLengthReplayBuffer(ReplayBufferInterface):
    
    def __init__(self, buffer_size: int = 5_000):
        self.buffer = list()  # initialise empty buffer
        self.buffer_size = 5_000

    def add_examples(self, examples: List[ReplayExample]):
        self.buffer.extend(examples)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]  # remove front (oldest) elements

    def get_examples(self, num_examples: int, random: bool = True):

        if num_examples > len(self.buffer):
            return []

        if random:
            return sample(list(self.buffer), num_examples)
        else:
            return list(self.buffer)[-num_examples:]

class HeuristicSortedReplayBuffer(FixedLengthReplayBuffer):

    # Replay buffer where the replay examples are sorted by some heuristic
    #  - Replay examples that have higher / lower heuristic (depending on if reverse sort or not)
    #    will be in memory for more time, thus they will influence model dynamics more.

    def __init__(self, reverse_sort: bool = False, buffer_size: int = 5_000):
        super().__init__(buffer_size)
        self.reverse_sort = reverse_sort

    def add_examples(self, examples: List[HeuristicReplayExample]):
        # add examples to buffer and sort based on hinge loss
        # remove examples at bottom of list if list exceeds maximum size
        self.buffer.extend(examples)
        self.buffer.sort(key = lambda ex: ex.h_value, reverse = self.reverse_sort)
        
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]  # remove examples with smallest / largest hinge loss (depending on if reversed)

class GeneralisedHeadModel(nn.Module):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        reward_space: gym.Space,
        replay_buffer: ReplayBufferInterface,
        device: str = 'cuda:0'
    ):
        super().__init__()

        image_shape = observation_space["x"].shape
        assert image_shape == (3, 28, 28), "this example only works on mnist-like data"
        assert isinstance(action_space, spaces.Discrete)
        assert action_space == reward_space
        self.num_classes = action_space.n
        image_channels = image_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # encodes features into internal representation
        # outputs of this module are fed into the SVM heads
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )

        # create a linear layer to represent linear SVM model
        # - svm has a single output, so number of output neurons is 1 for each 'svm'
        # - one svm needed for each class as we use the one-vs-rest approach to enable multi-class classification
        # - hence, we have 84 input features for each svm, and one output for each class / svm, so we get the below formulation. 
        self.head_layer : nn.Module = nn.Linear(84, self.num_classes)
        self.trainer: Trainer

        # create replay buffer
        self.replay_buffer = replay_buffer

        self.c = 1
        self.device = device

    def forward(self, x: Union[Tensor, Observations]) -> Tensor:
        # NOTE: here we don't make use of the task labels.
        if type(x) is Observations:
            x = x.x  # unpack x

        features = self.features(x)
        encoding = self.encoder(features)
        logits = self.head_layer(encoding)
        return logits

    def shared_step(
        self, batch: Tuple[Observations, Optional[Rewards]], environment: Environment
    ) -> Tuple[Tensor, Dict]:

        # Since we're training on a Passive environment, we will get both observations
        # and rewards, unless we're being evaluated based on our training performance,
        # in which case we will need to send actions to the environments before we can
        # get the corresponding rewards (image labels).

        observations: Observations = batch[0]
        rewards: Optional[Rewards] = batch[1]

        # get examples from replay
        replay_examples = self.replay_buffer.get_examples(32)

        if rewards is None:
            # If the rewards in the batch is None, it means we're expected to give
            # actions before we can get rewards back from the environment.
            rewards = environment.send(Actions(y_pred))

        x = observations.x
        gt_class = rewards.y

        if len(replay_examples) != 0:
            replay_x = torch.stack(tuple(ex.x for ex in replay_examples), dim=0)
            replay_y = torch.stack(tuple(ex.y for ex in replay_examples), dim=0)

            x = torch.cat((observations.x, replay_x.to(self.device)), dim=0)
            gt_class = torch.cat((rewards.y, replay_y.to(self.device)), dim=0).to(torch.int64)

        # Get the predictions:
        logits = self(x)
        y_pred = logits.argmax(-1)

        y_indices = gt_class
        y_targets = torch.nn.functional.one_hot(gt_class, num_classes=self.num_classes)
        y_targets[y_targets == 0] = -1  # need to set to -1 instead of 0 for SVM loss function

        accuracy = (y_pred == gt_class).float().sum() / len(gt_class)
        metrics_dict = {"accuracy": accuracy.item()}

        loss, batch_losses = self.batch_loss_fn(y_indices, y_targets, logits)

        # add new examples to replay
        examples = []
        for i, x in enumerate(observations.x):
            examples.append(HeuristicReplayExample(x.cpu(), rewards.y[i].cpu(), torch.mean(batch_losses[i]).cpu()))

        self.replay_buffer.add_examples(examples)

        return loss, metrics_dict

    def batch_loss_fn(self, y_indices, y_targets, logits):
        # Function that should return the loss for given targets (y_targets) and predictions (logits), on an individual basis
        # (each example in batch should have associated loss, in case this is needed for e.g. prioritised replay) as well as
        #  the averaged value
        raise NotImplementedError

class SVMHeadModel(GeneralisedHeadModel):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        reward_space: gym.Space,
        replay_buffer: ReplayBufferInterface,
        device: str = 'cuda:0'
    ):
        super().__init__(
            observation_space,
            action_space,
            reward_space,
            replay_buffer,
            device
        )

        # create a linear head layer to represent linear SVM model
        # - svm has a single output, so number of output neurons is 1 for each 'svm'
        # - one svm needed for each class as we use the one-vs-rest approach to enable multi-class classification
        # - hence, we have 84 input features for each svm, and one output for each class / svm, so we get the below formulation. 
        self.head_layer : nn.Module = nn.Linear(84, self.num_classes)

    def batch_loss_fn(self, y_indices, y_targets, logits):
        reg_loss = torch.mean(self.head_layer.weight.square() / 2)
        hinge_loss = torch.square(
            torch.max(
                torch.zeros(y_targets.shape).to(self.device), 1 - y_targets * logits
            )
        )
        loss = 0.01 * reg_loss + self.c * torch.mean(hinge_loss)

        return loss, hinge_loss

class SoftmaxHeadModel(GeneralisedHeadModel):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        reward_space: gym.Space,
        replay_buffer: ReplayBufferInterface,
        device: str = 'cuda:0'
    ):
        super().__init__(
            observation_space,
            action_space,
            reward_space,
            replay_buffer,
            device
        )

        # create a linear head layer 
        self.head_layer : nn.Module = nn.Linear(84, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def batch_loss_fn(self, y_indices, y_targets, logits):
        batch_losses = self.loss_fn(logits, y_indices)
        loss = torch.mean(batch_losses)

        return loss, batch_losses


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

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        """Adds command-line arguments for this Method to an argument parser."""
        parser.add_arguments(cls.HParams, "hparams")

    @classmethod
    def from_argparse_args(cls, args: Namespace):
        """Creates an instance of this Method from the parsed arguments."""
        hparams: cls.HParams = args.hparams
        return cls(hparams=hparams)


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


def demo_command_line():
    """ Run this quick demo from the command-line. """
    parser = ArgumentParser(description=__doc__)
    # Add command-line arguments for the Method and the Setting.
    SVMMethod.add_argparse_args(parser)
    # Add command-line arguments for the Setting and the Config (an object with
    # options like log_dir, debug, etc, which are not part of the Setting or the
    # Method) using simple-parsing.
    parser.add_arguments(DomainIncrementalSLSetting, "setting")
    parser.add_arguments(Config, "config")
    args = parser.parse_args()

    # Create the Method from the parsed arguments
    method: SVMMethod = SVMMethod.from_argparse_args(args)
    # Extract the Setting and Config from the args.
    setting: DomainIncrementalSLSetting = args.setting
    config: Config = args.config

    # Run the demo, applying that SVMMethod on the given setting.
    results: Results = setting.apply(method, config=config)
    print(results.summary())
    print(f"objective: {results.objective}")


if __name__ == "__main__":
    demo_simple()
