import sys
import os
import torch

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

sys.path.insert(0,f'{DIR}/../')

import nn

import pandas as pd
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
np.random.seed(4)
X, t = datasets.make_blobs(n_samples=100, centers=3, n_features=4, center_box=(0, 10))

"""
def test_1():
    import torch.nn as tnn

    torch_layer = tnn.Linear(4,10)
    W,b = list(torch_layer.parameters())
    W = W.detach().numpy().tolist()
    b = b.detach().numpy().tolist()
    solution = torch_layer(torch.Tensor(X[:4])).detach().numpy()
    
    layer = nn.Linear(4,10)
    layer.W = W
    layer.b = b
    answer = np.array(layer.transform(X[:4]))

    assert np.all(np.abs(solution-answer) <= 0.0001)
    
def test_2():
    import torch.nn as tnn
    # With square kernels and equal stride
    m = tnn.Conv2d(16, 2, 3, stride=1)
    solution = np.array(m.weight.shape)
    
    layer = nn.Conv2d(16, 2, 3)
    answer = np.array(np.array(layer.W).shape)
    
    assert np.all(np.abs(solution-answer) == 0)
    
def test_3():
    import torch.nn as tnn
    input = torch.randn(20, 16, 32, 32)
    # With square kernels and equal stride
    m = tnn.Conv2d(16, 2, 3, padding=1)
    solution = np.array(m(input).detach().numpy().shape)
    
    layer = nn.Conv2d(16, 2, 3)
    answer = np.array(np.array(layer.transform_dryrun(input)).shape)
    
    assert np.all(np.abs(solution-answer) == 0)
    
def test_4():
    import torch.nn as tnn
    input = torch.randn(20, 16, 32, 32)
    # With square kernels and equal stride
    m = tnn.Conv2d(16, 2, 3, padding=1,bias=False)
    
    solution = np.array(m(input).detach().numpy())
    
    layer = nn.Conv2d(16, 2, 3)
    layer.W = m.weight.detach().numpy().tolist()
    answer = np.array(layer.transform(input))
    
    assert np.all(np.abs(solution-answer) <= 1e-4)

def test_maxpool2d_shape():
    import torch.nn as tnn
    input = torch.randn(20, 16, 32, 32)
    m = tnn.MaxPool2d(kernel_size=3, stride=2)
    solution = np.array(m(input).detach().numpy().shape)
    
    layer = nn.MaxPool2d(kernel_size=3, stride=2)
    answer = np.array(np.array(layer.transform(input.numpy().tolist())).shape)
    
    assert np.all(np.abs(solution-answer) == 0)

def test_maxpool2d_values():
    import torch.nn as tnn
    input = torch.randn(10, 3, 32, 32)
    m = tnn.MaxPool2d(kernel_size=3, stride=2)
    solution = m(input).detach().numpy()
    
    layer = nn.MaxPool2d(kernel_size=3, stride=2)
    answer = np.array(layer.transform(input.numpy().tolist()))
    
    assert np.all(np.abs(solution-answer) <= 1e-6)

def test_batchnorm2d_shape():
    import torch.nn as tnn
    input = torch.randn(20, 16, 32, 32)
    m = tnn.BatchNorm2d(16)
    solution = np.array(m(input).detach().numpy().shape)
    
    layer = nn.BatchNorm2d(16)
    answer = np.array(np.array(layer.transform(input.numpy().tolist())).shape)
    
    assert np.all(np.abs(solution-answer) == 0)

def test_batchnorm2d_values():
    import torch.nn as tnn
    input = torch.randn(10, 3, 32, 32)
    m = tnn.BatchNorm2d(3)
    solution = m(input).detach().numpy()
    
    layer = nn.BatchNorm2d(3)
    answer = np.array(layer.transform(input.numpy().tolist()))
    
    assert np.all(np.abs(solution-answer) <= 1e-6)

def test_dropout_shape():
    import torch.nn as tnn
    input = torch.randn(20, 100)
    m = tnn.Dropout(p=0.5)
    solution = np.array(m(input).detach().numpy().shape)
    
    layer = nn.Dropout(p=0.5)
    answer = np.array(np.array(layer.transform(input.numpy().tolist())).shape)
    
    assert np.all(np.abs(solution-answer) == 0)

def test_dropout_nonzero():
    import torch.nn as tnn
    input = torch.ones(1000, 1000)
    m = tnn.Dropout(p=0.5)
    solution = np.count_nonzero(m(input).detach().numpy())
    
    layer = nn.Dropout(p=0.5)
    answer = np.count_nonzero(np.array(layer.transform(input.numpy().tolist())))
    
    # Allow for some variance due to randomness
    assert abs(solution - answer) / (1000 * 1000) < 0.05



def test_alexnet_shapes():
    # Create input tensor
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Initialize both models
    torch_alexnet = models.AlexNet()
    custom_alexnet = nn.AlexNet()
    
    # Expected shapes for each layer in features section
    expected_shapes = [
        (1, 64, 55, 55),    # Conv2d
        (1, 64, 55, 55),    # ReLU
        (1, 64, 27, 27),    # MaxPool
        (1, 192, 27, 27),   # Conv2d
        (1, 192, 27, 27),   # ReLU
        (1, 192, 13, 13),   # MaxPool
        (1, 384, 13, 13),   # Conv2d
        (1, 384, 13, 13),   # ReLU
        (1, 256, 13, 13),   # Conv2d
        (1, 256, 13, 13),   # ReLU
        (1, 256, 13, 13),   # Conv2d
        (1, 256, 13, 13),   # ReLU
        (1, 256, 6, 6),     # MaxPool
    ]
    
    # Test each layer
    x_custom = input_tensor.numpy().tolist()
    for i, layer in enumerate(custom_alexnet.features):
      if hasattr(layer, 'transform'):
          x_custom = layer.transform(x_custom)
      else:
          x_custom = layer(torch.tensor(x_custom)).detach().numpy().tolist()       
      custom_shape = np.array(x_custom).shape
      
      print(f"Layer {i}:")
      print(f"Expected shape: {expected_shapes[i]}")
      print(f"Got shape: {custom_shape}")
      
      assert custom_shape == expected_shapes[i], \
          f"Shape mismatch at layer {i}. Expected {expected_shapes[i]}, got {custom_shape}"
"""
def test_forward_pass():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    input_tensor = torch.randn(1, 3, 224, 224)
    custom_alexnet = nn.AlexNet()
    
    # Test forward pass
    output = custom_alexnet.forward(input_tensor.numpy().tolist())
    assert np.array(output).shape == (1, 1000), \
        "Final output shape should be (1, 1000)"[3]
