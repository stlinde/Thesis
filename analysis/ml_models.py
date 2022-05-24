# analysis/ml_models.py
"""
This module will implement the Neural Network Models used in the analysis.
"""
#IMPORT
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.input_size = input_size

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class EnsembleNeuralNetwork(nn.Module):
    def __init__(self):
        super(EnsembleNeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
        )

        self.har_stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 64)
        ) 

        self.ensemble = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        pass

    def forward(self, x, har_x):
        x = self.linear_relu_stack(x)
        har_x = self.har_stack(har_x)
        output = self.ensemble(x + har_x)
        return output


class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class TemporalConvolutionalHARNetwork(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
