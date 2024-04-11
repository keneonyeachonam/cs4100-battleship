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
from board import Board
from ship import Ship
import random

env = gym.make()


''' Generate game data to train on '''
board = Board(5)

def make_and_place_ships():
    # create a random seed?
    ships_positions = []

    for i in range(0, 3):
        size = random.randint(1, 3)
        row = random.randint(0, 4)
        col = random.randint(0, 4)

        new_ship = Ship(str("ship" + i), size)
        
        if (row, col) not in ships_positions:
            board.place_ship(new_ship, row, col)

        ships_positions.append((row, col))

def play_games(iter):
    game_boards = np.array([])

    for i in range(0, iter):
        while not board.check_gameover():
            row = random.randint(0, 4)
            col = random.randint(0, 4)
            board.missile(row, col)
    
        game_boards.append(board.AIBoard)

def generate_board_data(num_samples):
    boards = []
    labels = []  # This would ideally be the optimal next move based on historical data
    
    for _ in range(num_samples):
        board = np.random.randint(0, 2, (5, 5))  # Random board states - Making random ships across the board. Not our def of ships
        label = np.random.rand(25)               # Random 'optimal' move probabilities
        label /= label.sum()                     # Normalize to make it a probability distribution
        boards.append(board)
        labels.append(label)
        
    return torch.tensor(boards, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)




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

# Train model!
# num_epochs = 20
# loss_over_time = []

# for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     print(f"Training loss: {running_loss}")
#     loss_over_time.append(running_loss)

# print('Finished Training') 

