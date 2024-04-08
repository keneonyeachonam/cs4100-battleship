# Creating the board and its functions
import numpy as np
from enum import Enum
import random

from ship import Direction, pick_direction

HIT_EMPTY = -1
HIT_SHIP = 100
NONHIT_SHIP = 1
NONHIT_EMPTY = 0


class Board():

    '''
    Creating a board object

    Attrbutes:
    - size (int): the dimension of a square size x size board
    - ships (array) = array of Ship objects to be placed on board
    - num_fires (int) = numbers of moves made overall (hits or misses)

    Visible to AI
    - board_AI (array) = 2D array representing gameboard
        - 0 for ocean, -1 for miss, 100 for hit
    
    Hidden to AI ('answer key')
    - actual_board (not seen by AI) (array) = 2D array representing gameboard
        - 0 for ocean, 100 for actual ship locations
        - this doesn't change throughout the game, so we don't update/change for hits/misses

    Capabilites:
    - initalize_board = initalizes board (Board has list of ships but aren't placed yet)
    - place_ships = places ships on board (assigns ships random direction and random head coords)
        - CONSIDERATION: might be hard to not put ships out of bounds. do we want to stick to
          1-unit ships to avoid this issue?
    - hit = updates board (hit a ship)
    - miss = updates board (missed a ship)
    - num_ships = # of individual ships on board
    - num_occ = # of squares occupied by ships currently

    Behaviors:
    - guess(x, y)                           -- guesses a coordinate
    - missile() (coordinates : (int, int))  -- attempt a hit at given coordinate (GUESS?)
    - num_ships_left()                      -- give the numbers of ships un-sunk
    - score()                               -- updates current score (# tiles hit)
    '''

    # Define creation of a Board.
    def __init__(self, size): # takes in desired size and list of created ships
        self.size = size
        self.ships = []    # list of Ships to be placed on board
        self.num_fires = 0 # Initially the fire count is 0

        # Visible to AI, should be instantiated as a 2d array of 0s.
        # Updated when ships are actually placed
        self.AIBoard = np.zeros((size, size)) # -1 for miss, 100 for hit
        
        # Actual Board, should be instantiated as 0's, 100's where ships are located.
        # Instantiates as an empty board, then should add ships as provided.
        # Remains constant throughout entire game.
        # Updated when ships are actually placed AND when hits/misses are made
        self.HiddenBoard = np.zeros((size, size))

    

    def score(self): 
        '''
        Computes the calculation of the score of the board
        '''
        return np.sum(self.AIBoard)


    def num_misses(self):
        '''
        Returns how many misses have been made.
        If the value at each coordinate is equal to negative 1, add one to the count of misses.
        '''
        misses = 0
        for row in Board.AIBoard:
            for column in row:
                if Board.AIBoard[row][column] == -1:
                    misses =+ 1
        return misses
    


    def place_ship(self, ship, row, col):
        '''
        Adds ship and its coordinates to the board (hidden board)
        NONHIT_SHIP = part of a ship's positon on board - assuming it's more than 1 unit
        Return message when given coordinates cause ship to go off the board and adjusts its size
        '''
        ship.dir = pick_direction()

        self.ships.append([ship, (row, col)]) 

        place_count = 0
        exceeded_dim_message = f"{ship.name} has exceeded dimensions from {row}, {col}."

        if (ship.dir == Direction.NORTH):
            for i in range(0, ship.size):
                if (row + i < self.size):
                    self.HiddenBoard[row + i][col] = NONHIT_SHIP
                    place_count += 1
                else:
                    ship.size = place_count 
                    print(exceeded_dim_message + f" Size is now {ship.size}")
                    return
        elif (ship.dir == Direction.SOUTH):
            for i in range(0, ship.size):
                if (row - i >= 0):
                    self.HiddenBoard[row - i][col] = NONHIT_SHIP
                    place_count += 1
                else:
                    ship.size = place_count 
                    print(exceeded_dim_message + f" Size is now {ship.size}")
                    return
        elif (ship.dir == Direction.WEST):
            for i in range(0, ship.size):
                if (col + i < self.size):
                    self.HiddenBoard[row][col + i] = NONHIT_SHIP 
                    place_count += 1 
                else:
                    ship.size = place_count 
                    print(exceeded_dim_message + f" Size is now {ship.size}")
                    return
        elif (ship.dir == Direction.EAST):
            for i in range(0, ship.size):
                if (col - i >= 0):
                    self.HiddenBoard[row][col - i] = NONHIT_SHIP
                    place_count += 1
                else:
                    ship.size = place_count 
                    print(exceeded_dim_message + f" Size is now {ship.size}.")
                    return

        
    def missile(self, coordinates):
        """" 
        Launches a missle at the coordinate points

        Return -1 for a miss and return 100 for a hit
        AA missile launched at  location already fired returns a miss
        """
        moves = []

        x = coordinates[0]
        y = coordinates[1]

        ship_coors = [ship[1] for ship in self.ships]

        if (x, y) not in moves:
            # Add one to the total fires.
            # This should update misses and make the appropriate changes.
            if (x, y) in ship_coors:
                self.num_fires =+ 1
                self.HiddenBoard[x][y] = HIT_SHIP
            else:
                self.num_misses =+ 1
                self.HiddenBoard[x][y] = HIT_EMPTY


            # Make the reward at the hidden board visible on the AI board.
            # checks if given coordinate is a miss or hit, given if (x, y) is in list of tuples (ships)
            if (x, y) in ship_coors:
                self.AIBoard[x][y] = 100  # set AIBoard[x][y] to 100 if a ship was hit
                ship_idx = ship_coors.index((x, y))
                self.ships[ship_idx][0].num_hit += 1  
                self.ships[ship_idx][0].num_left -= 1
                moves.append((x, y))
                return 100
            else:
                self.AIBoard[x][y] = -1  # set AIBoard[x][y] to -1 if an empty tile was hit 
                moves.append((x, y))
                return -1
        else:
            raise("Move already made!")
        
        

    def check_gameover(self): 
        """ Check gameover condition (all ships sunk). """
        # ship[0] since each ship in ships is a list (ships is a list of lists)
        total_life = sum([ship[0].num_left for ship in self.ships])

        if total_life == 0: 
            return True 
        else: 
            return False 
        
    
    def num_ships_left(self):
        '''
        Returns the number of ships left on board
        '''
        afloat_ships = 0

        for ship in self.ships:
            if ship[0].num_left > 0:
                afloat_ships += 1

        return afloat_ships

        

        
