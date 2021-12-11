
from torch import nn

class BasicMNISTNetwork(nn.ModuleList):

    def __init__(self, n_channels, n_classes):
        modules = [
            # extracts features from input image using simple conv net
            nn.Sequential(
                nn.Conv2d(n_channels, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),

            # encodes features into internal representation
            # outputs of this module are fed into the SVM heads
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
            ),

            # head / classifying layer
            # either uses softmax or SVM classification depending upon algorithm being tested
            nn.Linear(84, n_classes)
        ]

        super().__init__(modules)