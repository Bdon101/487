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


def test_alexnet_output():
    import torchvision.models as models

    input = torch.randn(1, 3, 224, 224)
    torch_alexnet = models.AlexNet()
    solution = torch_alexnet(input).detach().numpy()
    
    custom_alexnet = nn.AlexNet()
    
    # Copy weights from torch model to custom model
    for i, layer in enumerate(custom_alexnet.features):
        if hasattr(layer, 'W'):
            layer.W = torch_alexnet.features[i].weight.detach().numpy().tolist()
            if hasattr(layer, 'bias'):
                layer.bias = torch_alexnet.features[i].bias.detach().numpy().tolist()
    
    for i, layer in enumerate(custom_alexnet.classifier):
        if hasattr(layer, 'W'):
            layer.W = torch_alexnet.classifier[i].weight.detach().numpy().tolist()
            layer.b = torch_alexnet.classifier[i].bias.detach().numpy().tolist()
    
    answer = np.array(custom_alexnet.forward(input.numpy().tolist()))
    
    assert np.allclose(solution, answer, atol=1e-4)
"""
def test_alexnet_features_shape():
    import torch
    import torchvision.models as models
    
    input = torch.randn(1, 3, 224, 224)
    torch_alexnet = models.AlexNet()
    solution = torch_alexnet.features(input).shape
    
    custom_alexnet = nn.AlexNet()
    x = input.numpy().tolist()
    for layer in custom_alexnet.features:
        if hasattr(layer, 'transform'):
            x = layer.transform(x)
        else:
            x = layer(torch.tensor(x)).detach().numpy().tolist()
    answer = np.array(x).shape
    
    assert np.all(np.abs(np.array(solution) - np.array(answer)) == 0)
