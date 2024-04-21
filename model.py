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
from torch.utils.data import DataLoader, TensorDataset
import math 

if torch.backends.mps.is_available():
	device=torch.device("mps")
elif torch.cuda.is_available():
	device=torch.device("cuda")
else:
	device=torch.device("cpu")


class Data(Dataset):
    '''
    Convert data to torch tensors and gets the trainloader/traindata and testloader/testdata
    '''
    def __init__(self, boards, labels):
        self.X = boards
        self.y = labels
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return len(self.X)
    
batch_size = 64


class Net(nn.Module):
    '''
    Our neural network model
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 25)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def val_format(x, y):
    '''
    Formats a given coordinate (x, y) into a single value (between 0 and 24)
    '''
    return x + (y * 5)

def play_game(board):
    '''
    Given a Board, plays the game of Battleship unitl game over
    Returns a 5 x 5 Numpy Array that represents the board and its total reward earned
    '''
    while not board.check_gameover():
        row = random.randint(0, 4)
        col = random.randint(0, 4)
        board.missile((row, col))

    return board.AIBoard


def play_game_record_moves(board):
    '''
    Given a start board, plays Battleship until game over and record the results
    '''
    game_data = []

    while not board.check_gameover(): # if game isn't over yet
        # select random move on board
        row = random.randint(0, 4)
        col = random.randint(0, 4)

        # save the state before the next move
        pre_move_gamestate = np.copy(board.AIBoard)
        
        # play move
        hit = board.missile((row, col))
        reward = 100 if hit else -1

        # save 
        game_data.append((pre_move_gamestate, val_format(row, col), reward))


    return game_data #  make move, return subsequent state


def make_and_place_ships():
    '''
    Places ships on board before gameplay begins
    '''
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


def augment_board(game_state):
    states = [game_state]
    # add rotations
    for k in range(1, 4):
        states.append(np.rot90(game_state, k))
    states.append(np.fliplr(game_state))  # flip left-right
    states.append(np.flipud(game_state))  # flip up-down
    return states


def generate_board_data(num_samples):
    '''
    num_samples: Number of games to play to create dataset
    Generate board data from a given amount of games
    '''
    boards = []
    labels = []  # actions stored as class labels
    rewards = []

    for _ in tqdm(range(num_samples)):
        board_with_ships = make_and_place_ships()
        game_data = play_game_record_moves(board_with_ships) # get game states
 
        total_misses = 0
        total_hits = 0

        for data in game_data:
            game_state = augment_board(data[0])[0].flatten() # flatten 5x5 board to a 25 element array
            action = data[1]
            reward = data[2]

            if reward == 100:
                total_hits += 1
            elif reward == 1:
                total_misses += 1

            boards.append(game_state.reshape(5, 5)) 
            labels.append(action)
            rewards.append(reward)

    boards_np = np.stack(boards)[:, None, :, :] 
    actions_np = np.array(labels, dtype=np.int64)

    return boards_np, actions_np


board, labels = generate_board_data(10000)

X_train, X_test, y_train, y_test = train_test_split(board, labels, test_size=0.33, random_state=42)
print("Splitting data")

# main training and testing model/data
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("TRAIN DATA: ", len(train_dataset))
print("TRAINLOADER: ", len(train_loader))

test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("TEST DATA: ", len(test_dataset))
print("TESTLOADER: ", len(test_loader))


def custom_loss(outputs, labels):
    labels = labels.long()
    gathered_probs = outputs[range(outputs.size(0)), labels]  # probability of selected actions
    log_probs = torch.log(gathered_probs.clamp(min=1e-9))
    return -torch.mean(log_probs)

net = Net().to(device)

# performs best with these parameters
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay = 0.01)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') # adjusts learning rate based on training processes
criterion = nn.CrossEntropyLoss()

# format a single value (0 - 24) into a coordinate (0-4, 0-4)
def coord_format(val):
    return (val % 5, np.floor_divide(val, 5))

def get_R(board):
    '''
    retrieves Reward, the value when guessed from hidden board
    '''
    R = board
    return R

num_epochs = 20
loss_over_time = []

for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
    running_loss = 0.0
    for inputs, labels in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        inputs = inputs.to(device).float()     
        labels = labels.to(device).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # channel dimension
        loss = criterion(outputs, labels) 

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step(running_loss)
    loss_over_time.append(running_loss / len(train_loader))
    print(f'Epoch {epoch+1}: Loss = {running_loss / len(train_loader)}')
    print(f"Training loss: {running_loss}")
    
print('Finished Training') 

# CHECKING ACCURACY:
total = 0
correct = 0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # total labels
        total += labels.size(0)
        # correctly predicted labels
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * (correct / total)} %')