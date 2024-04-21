from enum import Enum
import numpy as np
import random



class Direction(Enum):
    '''
    Direction on board. Points directly from one tile to another
    '''
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4
    
    
def pick_direction():
    '''
    Selects a random direction for a ship
    '''
    dir = random.randint(1, 4)
    
    if (dir == 1):
        return Direction.NORTH
    elif (dir == 2):
        return Direction.EAST
    elif (dir == 3):
        return Direction.SOUTH
    elif (dir == 4):
        return Direction.WEST
    

# Creating the ship and its functions. 

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
        self.name = name            # Name of Ship (String))
        self.size = size            # Size of Ship in units (int)
        self.body = [0] * size      # Body of Ship (array of 0's to start)
        self.dir = pick_direction() # orientation of ship (matters for board); None is its default before being placed on the board
        self.num_left = size        # number of unhit tiles left on ship
        self.num_hit = 0            # number of hit tiles left on ship
        self.isSunk = False;        # not sunk yet


    def get_name(self):
        '''
        Gets the name of the ship
        '''
        return self.name


    def size(self):
        '''
        Gets the size of the ship
        '''
        return self.size



    def hit(self):
        '''
        Check if ship is sunk and reduces the amount of ships left
        '''
        self.num_left -= 1  # decrement num_left
        self.num_hit += 1   # increment num_hit

        if (self.num_left == 0):
            self.isSunk = True
        
        return self.isSunk
    

    def num_left():
        '''
        Returns the number of ships left on the board
        '''
        return Ship.num_left



