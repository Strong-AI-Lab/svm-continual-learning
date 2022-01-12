
from typing import Dict, Any, List, Tuple, Type, Optional, Union, Callable
import itertools

from sequoia.common.config.config import Config
from sequoia.settings.base.results import Results
from sequoia.settings.sl.continual.setting import ContinualSLSetting

from svm.method.method import GenericMethod

import numpy as np

class BasicMethodTestingSuite():

    # Class that enables for the testing and comparison of multiple methods in a standardised
    # fashion.

    def __init__(self, method_init_fn: Callable[...,List[GenericMethod]], setting: ContinualSLSetting, config: Config, n_runs: int = 1):

        # stores the results of each method's execution
        # stored in same order as self.methods (i.e. first method results will be at front of list)
        # each method has a list of results, as there could be more than one run (depending on value of n_runs)
        self.results: List[List[Results]] = []

        # a callback function that returns the list of methods to test
        # (hacky solution to allowing multiple repeats of a methods execution, TODO: add reset function to method / model classes to make this cleaner)
        self.method_init_fn = method_init_fn

        # the setting that we are testing the methods on
        self.setting = setting

        # the config object
        self.config  = config

        # how many times we should run each method
        self.n_runs = n_runs

    def run(self):
        # runs the testing suite

        methods = self.method_init_fn()
        for method in methods:
            self.results.append([self.setting.apply(method, self.config)])

        all_methods = list(zip(methods))  # keep track of the separate instances for each method (zip with nothing to turn into list of tuples)
        for _ in range(1, self.n_runs):
            methods = self.method_init_fn()
            for method_i, method in enumerate(methods):
                self.results[method_i].append(self.setting.apply(method, self.config))
            all_methods = list(zip(all_methods, methods))
                                                                          

        # create return objects for each method, detailing evaluation stats among other things
        ret_objs: List[Dict] = []
        for method_i, results_arr in enumerate(self.results):
            accuracy_matrix = np.zeros((self.n_runs, self.setting.nb_tasks+1)) # store accuracy values for each task in matrix for easy mean / std dev calcs.
                                                                             # (+1 for task count to account for average task accuracy metric - see below *)
            for repeat_i, results in enumerate(results_arr):
                for task_i, task_metric in enumerate(results.final_performance_metrics):
                    accuracy_matrix[repeat_i][task_i] = task_metric.accuracy
                # * calculate and store average accuracy
                accuracy_matrix[repeat_i][-1] = np.mean(accuracy_matrix[repeat_i, :-1])

            # calculate mean / std dev for method's accuracy in each task
            means = np.mean(accuracy_matrix, axis=0)    # calculate mean over n_repeats axis
            std_devs = np.std(accuracy_matrix, axis=0)  # likewise with std devs

            ret_objs.append({
                'raw_results': results_arr,
                'method_instances': all_methods[method_i],
                'accuracy_matrix': accuracy_matrix,
                'task_mean_acc': means[:-1],
                'mean_acc': means[-1],
                'task_acc_std': std_devs[:-1],
                'acc_std': std_devs[-1]
            })
        
        return ret_objs

class GridSearchTestingSuite(BasicMethodTestingSuite):

    class Parameter():
        # class used to define a parameter to be used in a grid search

        def __init__(self, name: str, min: float, max:float, n_inter_vals: int):
            self.name = name  # name of this parameter

            self.min = min  # minimum value this parameter should take
            self.max = max  # maximum value this parameter should take

            self.n_inter_vals = n_inter_vals  # number of values within interval to test (by default, min and max are always tested)
            step_size = (max-min)/(n_inter_vals+1)
            self.vals = [min] + [min + i*step_size for i in range(1, n_inter_vals+1)] + [max]  # range is end-exclusive, so we add max at the end of the list manually

    def __init__(self, parameter_specs: List[Parameter], method_init_callback: Callable[..., GenericMethod], setting: ContinualSLSetting, config: Config, n_runs: int = 1):

        param_combos = list(itertools.product(*[param_spec.vals for param_spec in parameter_specs]))  # generate all possible combinations of parameters (i.e., grid of params to search)
        print(param_combos)

        def methods_creation_fn():

            methods: List[GenericMethod] = []
            for param_combo in param_combos:
                # create method by passing keyword args to method init callback
                hparams = {param_spec.name: param_combo[i] for i, param_spec in enumerate(parameter_specs)}
                method: GenericMethod = method_init_callback(**hparams)

                # update method hparams to make visible the hyper parameter combination for each method
                # (preserves pre-existing key-value pairs in method's hparam dictionary, i.e. doesn't overwrite existing entries)
                temp_method_hparams = method.hparams.copy()
                method.hparams.update(hparams)
                method.hparams.update(temp_method_hparams)

                methods.append(method)

            return methods

        super().__init__(methods_creation_fn, setting, config, n_runs)



