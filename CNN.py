import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from board import Board
from ship import Ship
import random


def make_and_place_ships():
    # create a random seed?
    ships_positions = []
    board = Board(5)

    for i in range(0, 3):
        size = random.randint(1, 3)
        row = random.randint(0, 4)
        col = random.randint(0, 4)

        new_ship = Ship(str("ship" + i), size)
        
        if (row, col) not in ships_positions:
            board.place_ship(new_ship, row, col)

        ships_positions.append((row, col))

    return board

def play_game(board):
    while not board.check_gameover():
        row = random.randint(0, 4)
        col = random.randint(0, 4)
        board.missile(row, col)

    return board.AIBoard


def generate_board_data(num_samples):
    boards = []
    labels = []  # This would ideally be the optimal next move based on historical data
    
    for _ in range(num_samples):
        board = play_game(make_and_place_ships())  # Random board states
        label = np.random.rand(25)               # Random 'optimal' move probabilities
        label /= label.sum()                     # Normalize to make it a probability distribution

        boards.append(board)
        labels.append(label)
        
    return torch.tensor(boards, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def CNN():
    array_size = 3
    output = np.zeros(25)

    # Convolution Layer
    # Create selection of 3 x 3 arrays to feed into CNN
    # Slide over board until all have been covered
    arrays = []
    for x in range(array_size):
        for y in range(array_size):
            addition = [(x, y), (x + 1, y), (x + 2, y), (x, y + 1), (x + 1, y + 1), (x + 2, y + 1), (x, y + 2), (x + 1, y + 2), (x + 2, y + 2)]
            arrays.append(addition)

    # Outputs this: 
    '''
    [[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
    [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)],
    [(0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4)],
    [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)], 
    [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)], 
    [(1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4)], 
    [(2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2)], 
    [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (4, 3)], 
    [(2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (4, 3), (2, 4), (3, 4), (4, 4)]]
    '''
    
    # ReLu layer
    # Need to calculate values for each 3 x 3

    # Pooling Layer
    # Map values of 3 x 3 s to 25 

    return output


# Select the highest valid coordinate to guess
def selection():
    # The 25 size vector
    output = CNN()

    # The coordinate at which the highest predicted value
    val = binarysearch(max(output), output)
    # Put into coordinate format
    ans = coord_format(val)

    # Ensure valid move at ans, if not, then find next highest valid solution.
    while ans:
        # select next highest value.
        val = binarysearch(max(output), output)
        ans = coord_format(val)

    # Return the coordinate selected
    return ans


def coord_format(val):
    return (val % 5, np.floor_divide(val, 5))