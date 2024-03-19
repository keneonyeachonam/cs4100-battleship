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
    else:
        return Direction.WEST

class Ship():

    '''
    Attributes of a ship:
    - Name (string) = name of the ship
    - Size (int) = length of the ship
    - Coordinates (int, int) = location of head of ship? or should be tuple of coordinates?
    - Direction (Enum) = direction the shape is facing on the board
    '''
            

    # Define creation of a ship
    def __init__(self, name, size):
        self.name = name
        self.size = size 
        x = random.randint(0, size+1)
        y = random.randint(0, size+1)
        self.coordinates = (x, y) # random? -- coords of ship head
        self.dir = pick_direction() # orientation of ship
        self.num_left = size    # number of unhit tiles left on ship
        self.num_hit = 0
        self.isSunk = False; # not sunk yet

    # What is the size of this ship?
    def size(self):
        return self.size
    
    # This ship has been hit
    def hit(self):
        # decrement num_left/increment num_hit
        self.num_left -= 1
        self.num_hit += 1

        if (self.num_left == 0):
            self.isSunk = True
        
        return self.isSunk



