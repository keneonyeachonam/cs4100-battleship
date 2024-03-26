import numpy as np
from enum import Enum
import random

# Creating the ship and its functions. 

class Direction(Enum):
    '''
    Direction on board. Points directly from one tile to another
    '''
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4

def pick_direction():
    dir = random.randint(1, 5)
    
    if (dir == 1):
        return Direction.NORTH
    elif (dir == 2):
        return Direction.EAST
    elif (dir == 3):
        return Direction.SOUTH
    elif (dir == 4):
        return Direction.WEST

class Ship():
    '''
    Attributes of a ship:
    - Name (string) = name of the ship
    - Size (int) = length of the ship
    - Body (array of 0's) = the ship itself, each unit represented by a 0
    - Coordinates (int, int) = location of head of ship? or should be tuple of coordinates?
    - Direction (Enum) = direction the shape is facing on the board
    '''
            

    # Define creation of a ship
    def __init__(self, name, size):
        self.name = name            # Name of Ship in string
        self.size = size            # Size of ship in int (how many units)
        self.body = [0] * size
        # (x, y) = head of ship
        self.dir = pick_direction() # orientation of ship (matters for board)
        self.num_left = size        # number of unhit tiles left on ship
        self.num_hit = 0            # number of hit tiles left on ship
        self.isSunk = False;        # not sunk yet


    # What is the size of this ship?
    def size(self):
        return self.size
    

    # this ship has been hit
    def hit(self):
        # need a way to figure out which 0 on the ship has been hit
        # change that 0 to a 10
        
        self.num_left -= 1  # decrement num_left
        self.num_hit += 1   # increment num_hit

        if (self.num_left == 0):
            self.isSunk = True
        
        return self.isSunk
    
    
    # returns tiles left to hit 
    def get_num_left():
        return Ship.num_left



