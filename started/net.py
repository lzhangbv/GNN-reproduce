"""
PyTorch: Getting Started
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 3 * 32 * 32 -> 6 * 28 * 28 -> 6 * 14 * 14
        x = self.pool(F.relu(self.conv2(x))) # 6 * 14 * 14 -> 16 * 10 * 10 -> 16 * 5 * 5
        x = x.view(-1, 16 * 5 * 5) # 16 * 5 * 5 -> 400 
        x = F.relu(self.fc1(x)) # 400 -> 120
        x = F.relu(self.fc2(x)) # 120 -> 85
        x = self.fc3(x) # 84 -> 10
        return x