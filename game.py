import numpy as np
import gym
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

env = gym.make()

'''
NOTES FROM MEETING - 040324
- use a neural network (CNN)
   - start with 2x2 kernel with the 5x5 board, then do 2x2 with the resulting 4x4 board (may need only one!)
   - flatten result + each input connects with 25 outputs in our CNN
   - each output corresponds to a cell in board
   - use hw 3 CNN as skeleton code (also tutorials online)
   - should be less than 10 lines of code
   - layers: Convultion layer ->  nn.Linear (9, 25) (not working? try number between 9 and 25) -> Softmax
   - keep track of total loss (L = -log(P(a | s) * R) and do backprogation to get min loss
- flatten board to a (25, 1) array 
- use softmax as last layer (pick max prob, if invalid pick next biggest probabilty)
'''


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 25, (2, 2))
        self.fc1 = nn.Linear(9, 25)
        self.m = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.m
        return x
        
net = Net()

# criterion = (L = -log(P(a | s) * R) and do backprogation to get min loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)




