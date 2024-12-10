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
    

class MaxPool2d:
    def __init__(self, kernel_size=3, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def transform(self, X):
        batch_size, channels, height, width = len(X), len(X[0]), len(X[0][0]), len(X[0][0][0])
        
        # Calculate output dimensions
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        # Initialize output
        out = [
            [
                [[0 for _ in range(out_width)] for _ in range(out_height)]
                for _ in range(channels)
            ]
            for _ in range(batch_size)
        ]

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        max_val = float('-inf')
                        for m in range(self.kernel_size):
                            for n in range(self.kernel_size):
                                h = i * self.stride + m
                                w = j * self.stride + n
                                if h < height and w < width:
                                    max_val = max(max_val, X[b][c][h][w])
                        out[b][c][i][j] = max_val
        print(np.array(np.array(output).shape))

        return out

import math
class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Initialize gamma and beta
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features

    def transform(self, input):
        # input shape: [batch_size, num_features, height, width]
        batch_size = len(input)
        height = len(input[0][0])
        width = len(input[0][0][0])
        
        # Calculate batch mean and variance
        batch_mean = [0.0] * self.num_features
        batch_var = [0.0] * self.num_features
        
        for c in range(self.num_features):
            channel_sum = 0.0
            channel_sq_sum = 0.0
            num_elements = batch_size * height * width
            
            for n in range(batch_size):
                for h in range(height):
                    for w in range(width):
                        value = input[n][c][h][w]
                        channel_sum += value
                        channel_sq_sum += value ** 2
            
            batch_mean[c] = channel_sum / num_elements
            batch_var[c] = (channel_sq_sum / num_elements) - (batch_mean[c] ** 2)
        
        # Normalize and apply affine transformation
        output = [[[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(self.num_features)] for _ in range(batch_size)]
        
        for n in range(batch_size):
            for c in range(self.num_features):
                for h in range(height):
                    for w in range(width):
                        normalized = (input[n][c][h][w] - batch_mean[c]) / math.sqrt(batch_var[c] + self.eps)
                        output[n][c][h][w] = self.gamma[c] * normalized + self.beta[c]

        print(np.array(np.array(output).shape))
    
        return output

import random
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def transform(self, input):
        if self.p == 0:
            return input

        output = []
        for sample in input:
            sample_output = []
            for value in sample:
                if random.random() > self.p:
                    sample_output.append(value / (1 - self.p))
                else:
                    sample_output.append(0)
            output.append(sample_output)
        return output

import torch.nn as nn
class AlexNet:
    def __init__(self):
        self.features = [
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        ]
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = [
            Dropout(p=0.5),
            Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(4096, 4096),
            nn.ReLU(inplace=True),
            Linear(4096, 1000),
        ]
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.ReLU):
                x = torch.tensor(x, dtype=torch.float32)
                x = layer(x)
                x = x.tolist()
            elif hasattr(layer, 'transform'):
                x = layer.transform(x)
            else:
                x = layer(x)
        
        x = self.avgpool(torch.tensor(x, dtype=torch.float32))
        x = torch.flatten(x, 1).tolist()  # Flatten all dimensions except batch

        for layer in self.classifier:
            if isinstance(layer, nn.ReLU):
                x = torch.tensor(x, dtype=torch.float32)
                x = layer(x)
                x = x.tolist()
            elif hasattr(layer, 'transform'):
                x = layer.transform(x)
            else:
                x = layer(x)
        return x