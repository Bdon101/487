import random
import numpy as np

import math

import torch.nn as nn
import torch

def get_new_random_weight():
    return random.random()

class Linear:
    def __init__(self,in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # initialize weights and biases randomly
        self.W = []
        self.b = []
        random.seed(42)
        self.init_parameters()
        
    def init_parameters(self):
        self.W = [[get_new_random_weight() for _ in range(self.in_features)] for _ in range(self.out_features)]
        if self.bias:
            self.b = [get_new_random_weight() for _ in range(self.out_features)]
        else:
            self.b = [0.0] * self.out_features 

    def transform(self,X):
        out = []
        for x in X:  # Iterate over each sample in the batch
          y = [sum(w_i * x_i for w_i, x_i in zip(w_row, x)) + b_i for w_row, b_i in zip(self.W, self.b)]
          out.append(y)
        return out
    
class Conv2d:
    def __init__(self,in_channels, out_channels, kernel_size=3, stride = 1, padding = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = [
          [
            [
              [get_new_random_weight() for _ in range(kernel_size)]
              for _ in range(kernel_size)
            ]
            for _ in range(in_channels)
          ]
          for _ in range(out_channels)
        ]
        random.seed(42)

    
    def transform_dryrun(self,X):
        batch_size, in_channels, height, width = len(X), len(X[0]), len(X[0][0]), len(X[0][0][0])
        padded_height = height + 2
        padded_width = width + 2
        padded_X = [
            [
                [[0 for _ in range(padded_width)] for _ in range(padded_height)]
                for _ in range(in_channels)
            ]
            for _ in range(batch_size)
        ]

        # copy original X into the padded version
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(height):
                    for j in range(width):
                        padded_X[b][c][i + 1][j + 1] = X[b][c][i][j]

       # Calculate output dimensions
        out_height = (padded_height - self.kernel_size) // self.stride + 1
        out_width = (padded_width - self.kernel_size) // self.stride + 1



        # Output will have shape (batch_size, out_channels, out_height, out_width)
        out = [
            [
                [[0 for _ in range(out_width)] for _ in range(out_height)]
                for _ in range(self.out_channels)
            ]
            for _ in range(batch_size)
        ]
                # Perform convolution
        for b in range(batch_size):  # Over each batch
            for o in range(self.out_channels):  # Over each output channel
                for i in range(out_height):  # Over each output row
                    for j in range(out_width):  # Over each output column
                        # Apply the kernel to the input slice
                        result = 0
                        for c in range(self.in_channels):  # Over each input channel
                            for ki in range(self.kernel_size):  # Over kernel rows
                                for kj in range(self.kernel_size):  # Over kernel columns
                                    i_padded = i * self.stride + ki
                                    j_padded = j * self.stride + kj
                                    result += (
                                        padded_X[b][c][i_padded][j_padded]
                                        * self.W[o][c][ki][kj]
                                    )
                        out[b][o][i][j] = result
        return out
    
    def transform(self,X):
        out = self.transform_dryrun(X) 
        # Your solution here
        return out
    
