import numpy as np
import random

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

        # MOVED BEHAVIOR TO BOARD:
        # (x, y) = head of ship
        # MOVED TO BOARD: 
        self.dir = None             # orientation of ship (matters for board); None is its default before being placed on the board
        self.num_left = size        # number of unhit tiles left on ship
        self.num_hit = 0            # number of hit tiles left on ship
        self.isSunk = False;        # not sunk yet

    # What is the name of this ship?
    def get_name(self):
        return self.name

    # What is the size of this ship?
    def size(self):
        # self.body.size would also work(?)
        return self.size


    # I think this is all board behavior?
    # this ship has been hit
    def hit(self):
        # need a way to figure out which 0 on the ship has been hit
        # change that 0 to a 10
        
        self.num_left -= 1  # decrement num_left
        self.num_hit += 1   # increment num_hit

        if (self.num_left == 0):
            self.isSunk = True
        
        return self.isSunk
    

    def num_left():
        return Ship.num_left



