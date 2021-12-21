import torch

class SharedStepContext():

    # Wrapper class for data that is relevant to a given shared_step execution
    #  - Primarily used for replay purposes to enable easier passing of information related to
    #    calculating replay heuristics

    def __init__(self, ex_i: int, x: torch.Tensor, y_targets: torch.Tensor, y_pred: torch.Tensor, 
                 batch_losses: torch.Tensor, loss: torch.Tensor):
                 
        self.ex_i = ex_i
        self.x = x
        self.y_targets = y_targets
        self.y_pred = y_pred
        self.batch_losses = batch_losses
        self.loss = loss