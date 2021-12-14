from torch import nn
from copy import deepcopy
import numpy as np

from method import GenericMethod

from sequoia.settings.sl.incremental.results import Results

def copy_module_list(module_list: nn.ModuleList):
    # Returns a copy of a module list

    # NOTE: does deepcopy really work?
    new_copy = nn.ModuleList()
    for module in module_list:
        new_copy.append(deepcopy(module))

    return new_copy

def print_method_results(results: Results, method: GenericMethod):
    # Prints accuracy results of running method in a readable format
    #  - method_str: a str describing the nature of the method

    print(f'\n"{method.description}" results (final):')
    for i, metric in enumerate(results.final_performance_metrics):
        print(f' - Task {i} accuracy across {metric.n_samples} samples: {metric.accuracy}')
    print(f' - Average accuracy: {np.mean([metric.accuracy for metric in results.final_performance_metrics])}')