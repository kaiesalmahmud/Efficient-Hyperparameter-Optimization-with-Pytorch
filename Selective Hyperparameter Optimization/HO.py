import matplotlib.pyplot as plt
import numpy as np
import math
# %matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

from MNISTConvNet import MNISTConvNet
from nas_wot import score

dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                        name='learning_rate')

dim_num_conv_layers = Integer(low=1, high=3, name='name_conv_layers')

dim_num_fc_units = Integer(low=5, high=512, name='num_fc_units')

dim_kernel_size = Integer(low=3, high=5, name='kernel_size')

dimensions = [dim_learning_rate,
              dim_num_conv_layers,
              dim_num_fc_units,
              dim_kernel_size]

default_parameters = [1e-5, 1, 16, 3]


train_set = torchvision.datasets.MNIST(
                        root='./data/MNIST',
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(0, 1)
                        ])
)

test_set = torchvision.datasets.MNIST(
                        root='./data/MNIST',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(0, 1)
                        ])
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10
)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=10
)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train(model, lr, num_epoch, train_loader, test_loader):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch
            
            preds = model(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print('epoch:', epoch, 'total_correct:', total_correct, 'loss:', total_loss)

    print('Train Accuracy:', total_correct/len(train_set))

    test_loss = 0
    test_correct = 0

    for batch in test_loader:
        images, labels = batch

        preds = model(images)
        loss = F.cross_entropy(preds, labels)
        
        test_loss += loss.item()
        test_correct += get_num_correct(preds, labels)

    print('Test Accuracy:', test_correct/len(test_set))

    return test_correct/len(test_set)

best_accuracy = 0.0
best_model_path = './best_model.pt'

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_conv_layers,
            num_fc_units, dropout_rate):

    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_conv_layers:', num_conv_layers)
    print('num_fc_units:', num_fc_units)
    print('dropout_rate:', dropout_rate)
    
    model = MNISTConvNet(num_conv_layers=num_conv_layers,
                         num_fc_units=num_fc_units,
                         dropout_rate=dropout_rate)

    
    
    accuracy = train(model, learning_rate, 5, train_loader, test_loader)

    print('Accuracy:', accuracy)

    global best_accuracy

    if accuracy > best_accuracy:
        torch.save(model.state_dict(), best_model_path)
        best_accuracy = accuracy

    del model

    return -accuracy

search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=12,
                            x0=default_parameters)