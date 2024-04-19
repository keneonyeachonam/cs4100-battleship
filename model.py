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
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 25)
# OLD:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)  # output: 1x4x4 from 1x5x5 input
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)  # output: 32?
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(32 * 3 * 3, 25)  # should first param be 16 or 1024, 25 to 5
#         # self.softmax = nn.Softmax(dim=1)
#         # self.fc_out = nn.Linear(25, 25)  # Added this layer to produce logits for CrossEntropyLoss


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# OLD:
#         x = F.relu(self.conv1(x))  # after convolution
#         x = F.relu(self.conv2(x)) 
#         x = self.dropout1(x) 
#         x = torch.flatten(x, 1)  # flatten 4x4 grid to 16 element vector
#         x = self.fc1(x)  # fully connected layer
#         # x = self.softmax(x)  # softmax to get probabilities of each move
#         return x


def val_format(x, y):
    '''
    Format a coordinate (x, y) into a single value (0-24)
    '''
    return x + (y * 5)

def play_game(board):
    while not board.check_gameover():
        row = random.randint(0, 4)
        col = random.randint(0, 4)
        board.missile((row, col))

    return board.AIBoard


def play_game_record_moves(board):
    '''
    Given a start board, plays Battleship until game over and record the results
    '''
    # start_state = np.copy(board.AIBoard) # starting game state
    game_data = []

    while not board.check_gameover(): # if game isn't over yet
        # select random move
        row = random.randint(0, 4)
        col = random.randint(0, 4)

        # save state pre move
        pre_move_gamestate = np.copy(board.AIBoard)
        
        # play move
        hit = board.missile((row, col))
        reward = 100 if hit else -1

        # print(f"board: {pre_move_gamestate}")

        # save 
        game_data.append((pre_move_gamestate, val_format(row, col), reward))
    # state, cell moved on, reward
    # print("GAME DATA: ", game_data)
    return game_data #  make move, return subsequent state


def make_and_place_ships():
    # create a random seed?
    '''
    Places ships on board
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
    # Add rotations
    for k in range(1, 4):
        states.append(np.rot90(game_state, k))
    states.append(np.fliplr(game_state))  # Flip left-right
    states.append(np.flipud(game_state))  # Flip up-down
    return states


def generate_board_data(num_samples):
    boards = []
    labels = []  # actions stored as class labels
    rewards = []
    board_data = []

    # all_game_data = []
    for _ in tqdm(range(num_samples)):
        board_with_ships = make_and_place_ships()
        game_data = play_game_record_moves(board_with_ships) # get game states
        #all_game_data.extend(game_data)
        #board = play_game(make_and_place_ships())  # random board states
 
        total_misses = 0
        total_hits = 0

        for data in game_data:
            # print("HERE IS THE DATA: ", data)
            game_state = augment_board(data[0])[0].flatten() # flatten 5x5 board to a 25 element array
            # print("STATE: ", game_state)  
            action = data[1]
            # print("ACTION: ", action)
            reward = data[2]

            if reward == 100:
                total_hits += 1
            elif reward == 1:
                total_misses += 1

            # print("REWARD ", reward)
            # np.append(boards, game_state)
            # np.append(labels, action)
            # np.append(rewards, reward)
            boards.append(game_state.reshape(5, 5)) 
            labels.append(action)
            rewards.append(reward)


            # hits, miss, # of states, total score,
            # board_data.append([sum(rewards), len(boards), total_hits, total_misses]) 

            # boards.append()
            # label = np.zeros(25) 
            # label[move] = 1  # one hot encoding of the move made on that board
            # labels.append(label)
    
    # print(f"board: {boards}")
    # print(f"labels: {labels}")

    #boards_np = np.array(boards, dtype=np.float32)
    boards_np = np.stack(boards)[:, None, :, :] 
    # boards_np = np.stack(np.array(board_data))[:, None, :]
    actions_np = np.array(labels, dtype=np.int64)

    return boards_np, actions_np

# def generate_board_data(num_samples):
#     boards = []
#     labels = []  # This would ideally be the optimal next move based on historical data
    
#     for _ in range(num_samples):
#         board = play_game(make_and_place_ships())  # get game states and actions
    
#         label = np.random.rand(25)   # Random 'optimal' move probabilities
#         label /= label.sum()   # Normalize to make it a probability distribution

#         boards.append(board)
#         labels.append(label)
        
#     return torch.tensor(boards, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

board, labels = generate_board_data(10000) # 1000 games (10 for now)

X_train, X_test, y_train, y_test = train_test_split(board, labels, test_size=0.33, random_state=42)
print("Spliting data")


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
    gathered_probs = outputs[range(outputs.size(0)), labels]  # Probability of selected actions
    log_probs = torch.log(gathered_probs.clamp(min=1e-9))
    return -torch.mean(log_probs)

net = Net().to(device)

# correct = []
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') # adjusts learning rate based on training processes

# criterion = (L = -log(P(a | s) * R) and do backprogation to get min loss
criterion = nn.CrossEntropyLoss()

# format a single value (0 - 24) into a coordinate (0-4, 0-4)
def coord_format(val):
    return (val % 5, np.floor_divide(val, 5))

# criterion = (L = -log(P(a | s) * R) and do backprogation to get min loss

# (L = -log(P(a | s) * R)
# def criterion(x, y, board):
#     # Reward is the value when guessed. From hidden board
#     R = board.HiddenBoard[x][y]
#     ans = np.log(prob_action(board) * R) * -1
#     return ans

def get_R(board):
    # Reward is the value when guessed. From hidden board
    R = board
    return R

num_epochs = 50
loss_over_time = []

for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
    running_loss = 0.0
    #for i, data in enumerate(tqdm(train_dataloader, 0)):
    for inputs, labels in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        #inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.to(device).float()      #unsqueeze(1)
        labels = labels.to(device).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # channel dimension
        loss = criterion(outputs, labels)  # torch.dot(outputs, R)

        # loss = -1 * math.log(torch.matmul(outputs, get_R(board)))
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
# Accuracy: 3.608923884514436%
# Accuracy: 4.076436239885419%