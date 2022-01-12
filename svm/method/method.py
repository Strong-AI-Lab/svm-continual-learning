
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Type, Optional, Union, Callable

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

from svm.models.model import ClassificationModel

from collections import deque
from random import sample

class GenericMethod(Method, target_setting=IncrementalSLSetting): 

    @dataclass
    class HParams:
        """ Hyper-parameters of the method. """

        # Learning rate of the optimizer.
        learning_rate: float = 0.001

    def __init__(self, model: ClassificationModel, description: str = "Generic method", n_epochs: int = 1, hparams: Dict[str, Any]={}):
        self.hparams: Dict[str, Any] = hparams  # hyper-parameters of the method
        self.n_epochs: int = n_epochs
        self.early_stop_patience: int = 2

        self.model = model
        self.optimizer: torch.optim.Optimizer

        self.description = description

    def configure(self, setting: IncrementalSLSetting):
        """ 
            Called before the method is applied on a setting (before training). 
        """

        self.model.to(setting.config.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams['learning_rate'] if 'learning_rate' in self.hparams else 0.001,
        )

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Example train loop.
        You can do whatever you want with train_env and valid_env here.
        
        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # configure() will have been called by the setting before we get here.

        for epoch in range(self.n_epochs):
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

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        with torch.no_grad():
            logits = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    def on_task_switch(self, task_id):
        pass