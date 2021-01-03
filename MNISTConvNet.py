import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class MNISTConvNet(torch.nn.Module):
    def __init__(self, num_conv_layers, num_fc_units, dropout_rate, kernel_size=3, num_filters_1=3, num_filters_2=5, num_filters_3=7):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, num_filters_1, kernel_size=kernel_size)
        self.conv2 = None
        self.conv3 = None
        
        output_size = (28 - kernel_size + 1) // 2  # Q
        num_output_filters = num_filters_1
        
        if num_conv_layers > 1:
            self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=kernel_size)
            num_output_filters = num_filters_2
            output_size = (output_size - kernel_size + 1) // 2
            
        if num_conv_layers > 2:
            self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=kernel_size)
            num_output_filters = num_filters_3
            output_size = (output_size - kernel_size + 1) // 2
            
        self.dropout = nn.Dropout(p = dropout_rate)
        
        self.conv_output_size = num_output_filters * output_size * output_size
        
        self.fc1 = nn.Linear(self.conv_output_size, num_fc_units)
        self.fc2 = nn.Linear(num_fc_units, 10)
        
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        
        if not self.conv2 is None:
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        if not self.conv3 is None:
            x = F.max_pool2d(F.relu(self.conv3(x)), 2)
            
        x = self.dropout(x)
        
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
        
    
    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))


network = MNISTConvNet(num_conv_layers=3,
                       num_filters_1=3,
                       num_filters_2=5,
                       num_filters_3=7,
                       dropout_rate=0.01,
                       num_fc_units=120,
                       kernel_size=3)