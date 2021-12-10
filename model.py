
from abc import abstractmethod
from typing import Dict, List, Tuple, Type, Optional, Union

import gym
import torch
import numpy as np
from gym import spaces
from torch import Tensor, nn

from sequoia.settings import Environment
from sequoia.settings.sl.incremental.objects import (
    Actions,
    Observations,
    Rewards,
)

from replay import *

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

        y_targets = gt_class

        accuracy = (y_pred == gt_class).float().sum() / len(gt_class)
        metrics_dict = {"accuracy": accuracy.item()}

        loss, batch_losses = self.batch_loss_fn(y_targets, logits)

        # add new examples to replay
        examples = []
        for i, x in enumerate(observations.x):
            examples.append(HeuristicReplayExample(x.cpu(), rewards.y[i].cpu(), torch.mean(batch_losses[i]).cpu()))

        self.replay_buffer.add_examples(examples)

        return loss, metrics_dict

    @abstractmethod
    def batch_loss_fn(self, y_targets: Tensor, logits: Tensor) -> Tuple[Tensor, Tensor]:
        # Function that should return the loss for given targets (y_targets) and predictions (logits), on an individual basis
        # (each example in batch should have associated loss, in case this is needed for e.g. prioritised replay) as well as
        # the averaged value
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

    def batch_loss_fn(self, y_targets: Tensor, logits: Tensor) -> Tuple[Tensor, Tensor]:

        y_onehot = torch.nn.functional.one_hot(y_targets, num_classes=self.num_classes)
        y_onehot[y_onehot == 0] = -1  # need to set to -1 instead of 0 for SVM loss function

        reg_loss = torch.mean(self.head_layer.weight.square() / 2)
        hinge_loss = torch.square(
            torch.max(
                torch.zeros(y_onehot.shape).to(self.device), 1 - y_onehot * logits
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

    def batch_loss_fn(self, y_targets: Tensor, logits: Tensor) -> Tuple[Tensor, Tensor]:
        batch_losses = self.loss_fn(logits, y_targets)
        loss = torch.mean(batch_losses)

        return loss, batch_losses