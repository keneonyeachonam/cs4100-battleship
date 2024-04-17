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
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)  # output: 32?
        self.fc1 = nn.Linear(32 * 3 * 3, 25)  # should first param be 16 or 1024, 25 to 5
        # self.softmax = nn.Softmax(dim=1)
        # self.fc_out = nn.Linear(25, 25)  # Added this layer to produce logits for CrossEntropyLoss


    def forward(self, x):
        x = F.relu(self.conv1(x))  # after convolution
        x = F.relu(self.conv2(x))  
        x = torch.flatten(x, 1)  # flatten 4x4 grid to 16 element vector
        x = self.fc1(x)  # fully connected layer
        # x = self.softmax(x)  # softmax to get probabilities of each move
        return x

# format a coordinate (x, y) into a single value (0-24)
def val_format(x, y):
    return x + (y * 5)

def play_game(board):
    while not board.check_gameover():
        row = random.randint(0, 4)
        col = random.randint(0, 4)
        board.missile((row, col))

    return board.AIBoard

def play_game_record_moves(board):
    start_state = np.copy(board.AIBoard) # starting game state
    game_data = [start_state]

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
        
    return game_data #  make move, return subsequent state

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

def generate_board_data(num_samples):
    boards = np.array([])
    labels = np.array([])  # actions stored as class labels
    rewards = np.array([])

    # all_game_data = []
    for _ in tqdm(range(num_samples)):
        board_with_ships = make_and_place_ships()
        game_data = play_game_record_moves(board_with_ships) # get game states
        #all_game_data.extend(game_data)
        #board = play_game(make_and_place_ships())  # random board states

        for data in game_data:
            state = data[0]
            action = data[1]
            reward = data[2]
            boards = np.append(boards, state.flatten())
            labels = np.append(labels, action)
            rewards = np.append(rewards, reward)

            # boards.append()
            # label = np.zeros(25) 
            # label[move] = 1  # one hot encoding of the move made on that board
            # labels.append(label)

    boards_np = np.array(boards)
    actions_np = np.array(labels, dtype=np.float32)

    # print(f"boards_np: {boards}")

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

board, labels = generate_board_data(1000) # 1000 games

X_train, X_test, y_train, y_test = train_test_split(board, labels, test_size=0.33, random_state=42)
print("Spliting data")

# main training and testing model/data
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def custom_loss(outputs, labels):
    labels = labels.long()
    gathered_probs = outputs[range(outputs.size(0)), labels]  # Probability of selected actions
    log_probs = torch.log(gathered_probs.clamp(min=1e-9))
    return -torch.mean(log_probs)

net = Net().to(device)

# correct = []
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# criterion = (L = -log(P(a | s) * R) and do backprogation to get min loss
criterion = nn.CrossEntropyLoss()

num_epochs = 20
loss_over_time = []

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

print("started training")
total_loss = 0.0 

for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
    running_loss = 0.0
    #for i, data in enumerate(tqdm(train_dataloader, 0)):
    for inputs, labels in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        #inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.to(device).unsqueeze(1)
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

    print(f"Training loss: {running_loss}")
    loss_over_time.append(running_loss)

print('Finished Training') 
print(f'Epoch {epoch+1}: Loss = {running_loss / len(train_loader)}')



"""
We're not training so we don't need to calculate the gradients for our outputs
"""
total = 0
correct = 0
with torch.no_grad():
    for data in test_dataloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test instances: {100 * correct // total}%')