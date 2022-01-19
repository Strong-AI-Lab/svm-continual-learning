
from abc import abstractmethod
import re
from typing import Dict, List, Tuple, Type, Optional, Union, TYPE_CHECKING

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

from svm.replay.context import ModelPredictionContext
from svm.replay.buffer.base import AbstractReplayBuffer    

class ClassificationModel(nn.Module):

    def __init__(
        self,
        modules: nn.ModuleList,
        replay_buffer: AbstractReplayBuffer,
        device: str = 'cuda:0'
    ):
        super().__init__()

        self.module_list = modules  # Stores the modules of this model in a list-like format
        self.replay_buffer = replay_buffer
        self.replay_buffer.attach(self)
        self.device = device

    def forward(self, x: Union[Tensor, Observations]) -> Tensor:

        if type(x) is Observations:
            x = x.x  # unpack x

        for module in self.module_list:
            x = module(x)

        return x

    def shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], environment: Environment, use_replay: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:

        x = batch[0]
        y_targets = batch[1]

        # get examples from replay to supplement immediate observations if specified
        if use_replay:
            replay_examples = self.replay_buffer.get_examples(32)

            if len(replay_examples) != 0:
                replay_x = torch.stack(tuple(ex.x for ex in replay_examples), dim=0)
                replay_y = torch.stack(tuple(ex.y for ex in replay_examples), dim=0)

                x = torch.cat((x, replay_x.to(self.device)), dim=0)
                y_targets = torch.cat((y_targets, replay_y.to(self.device)), dim=0).to(torch.int64)

        # Get the predictions:
        y_pred = self(x)
        pred_class = y_pred.argmax(-1)

        accuracy = (pred_class == y_targets).float().sum() / len(y_targets)
        metrics_dict = {"accuracy": accuracy.item()}

        loss, batch_losses = self.batch_loss_fn(y_targets, y_pred)

        if use_replay:
            splitting_i = None  # None defaults to no splitting as x[:None] becomes x[:]
            if len(replay_examples) != 0:
                splitting_i = len(x) - len(replay_examples)

            # add new examples to replay buffer
            # TODO: should we calculate separate combined loss for replay / new examples?
            new_examples_context = ModelPredictionContext(-1, x[:splitting_i], y_targets[:splitting_i], y_pred[:splitting_i], batch_losses[:splitting_i], loss) 
            self.replay_buffer.add_examples(new_examples_context)

            if splitting_i is not None:
                # update examples that were pulled from the replay buffer
                replay_examples_context = ModelPredictionContext(-1, x[splitting_i:], y_targets[splitting_i:], y_pred[splitting_i:], batch_losses[splitting_i:], loss)
                self.replay_buffer.update_examples(replay_examples, replay_examples_context)

        return y_pred, loss, batch_losses, metrics_dict

    @abstractmethod
    def batch_loss_fn(self, y_targets: Tensor, logits: Tensor) -> Tuple[Tensor, Tensor]:
        # Function that should return the loss for given targets (y_targets) and predictions (logits), on an individual basis
        # (each example in batch should have associated loss, in case this is needed for e.g. prioritised replay) as well as
        # the averaged value
        raise NotImplementedError

    def on_task_switch(self, task_id: Optional[int]):
        self.replay_buffer.on_task_switch(task_id)

class SoftmaxClassificationModel(ClassificationModel):       

    def batch_loss_fn(self, y_targets: Tensor, logits: Tensor) -> Tuple[Tensor, Tensor]:
        batch_losses = nn.functional.cross_entropy(logits, y_targets, reduction='none')  # reduction = 'none' means we get individual losses
        loss = torch.mean(batch_losses)

        return loss, batch_losses

class SVMClassificationModel(ClassificationModel):

    def __init__(
        self,
        modules: nn.ModuleList,
        replay_buffer: AbstractReplayBuffer,
        c: float = 1,
        device: str = 'cuda:0'
    ):

        super().__init__(
            modules,
            replay_buffer,
            device
        )

        assert type(modules[-1]) is nn.Linear  # must be linear layer for linear SVM classification to function

        # Multiplicative constant for SVM hinge loss
        self.c = c

    def batch_loss_fn(self, y_targets: Tensor, logits: Tensor) -> Tuple[Tensor, Tensor]:

        head_layer = self.module_list[-1]  # head layer should be last module in modules
        num_classes = head_layer.weight.shape[0]

        y_onehot = torch.nn.functional.one_hot(y_targets, num_classes=num_classes)
        y_onehot[y_onehot == 0] = -1  # need to set to -1 instead of 0 for SVM loss function

        reg_loss = torch.mean(head_layer.weight.square() / 2)
        hinge_loss = torch.square(
            torch.max(
                torch.zeros(y_onehot.shape).to(self.device), 1 - y_onehot * logits
            )
        )
        loss = reg_loss + self.c * torch.mean(hinge_loss)

        return loss, hinge_loss