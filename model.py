import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from board import Board
from ship import Ship

import random
from sklearn.model_selection import train_test_split

# Convert data to torch tensors + gets the trainloader/traindata and testloader/testdata
class Data(Dataset):
    def __init__(self, boards, labels):
        self.X = boards
        self.y = labels
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return len(self.X)
    
batch_size = 64

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)  # output: 1x4x4 from 1x5x5 input
        self.fc1 = nn.Linear(16, 25)  # should first param be 16 or 1024
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # after convolution
        x = torch.flatten(x, 1)  # flatten 4x4 grid to 16 element vector
        x = self.fc1(x)  # fully connected layer
        x = self.softmax(x)  # softmax to get probabilities of each move
        return x

def generate_board_data(num_samples):
    boards = []
    labels = []  # This would ideally be the optimal next move based on historical data
    
    for _ in range(num_samples):
        board = play_game(make_and_place_ships())  # Random board states
        label = np.random.rand(25)   # Random 'optimal' move probabilities
        label /= label.sum()   # Normalize to make it a probability distribution

        boards.append(board)
        labels.append(label)
        
    return torch.tensor(boards, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def make_and_place_ships():
    # create a random seed?
    ships_positions = []
    board = Board(5)

    for i in range(0, 3):
        size = random.randint(1, 3)
        row = random.randint(0, 4)
        col = random.randint(0, 4)

        new_ship = Ship(str("ship" + str(i)), size)
        
        if (row, col) not in ships_positions:
            board.place_ship(new_ship, row, col)

        ships_positions.append((row, col))

    return board


def play_game(board):
    while not board.check_gameover():
        row = random.randint(0, 4)
        col = random.randint(0, 4)
        board.missile((row, col))

    return board.AIBoard

# format a single value (0 - 24) into a coordinate (0-4, 0-4)
def coord_format(val):
    return (val % 5, np.floor_divide(val, 5))

# format a coordinate (x, y) into a single value (0-24)
def val_format(x, y):
    return x + (y * 5)

# criterion = (L = -log(P(a | s) * R) and do backprogation to get min loss

# P(a | s)
def prob_action(board):
    # Should look at the prob for that action (# of valid) given state
    # action probability is the same for all actions possible from one state as long as they are valid
    # In this case the possible moves are marked as 0.
    possibles = np.sum(board == 0)
    return np.divide(1, possibles)

# (L = -log(P(a | s) * R)
# def criterion(x, y, board):
#     # Reward is the value when guessed. From hidden board
#     R = board.HiddenBoard[x][y]
#     ans = np.log(prob_action(board) * R) * -1
#     return ans
    
# main training and testing model/data
board, labels = generate_board_data(1000)   
model = Net()
ce = nn.CrossEntropyLoss()
X_train, X_test, y_train, y_test = train_test_split(board, labels, test_size=0.33, random_state=42)

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# criterion = (L = -log(P(a | s) * R) and do backprogation to get min loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train model!
num_epochs = 20
loss_over_time = []

for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = ce(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Training loss: {running_loss}")
    loss_over_time.append(running_loss)

print('Finished Training') 


"""
We're not training so we don't need to calculate the gradients for our outputs
"""
total = 0
correct = 0
with torch.no_grad():
    for data in test_dataloader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')