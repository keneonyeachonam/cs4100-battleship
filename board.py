# Creating the board and its functions
import numpy as np

from ship import Direction

HIT_EMPTY = 'X'
HIT_SHIP = 'O'
NONHIT_SHIP = '_'
NONHIT_EMPTY = '_'

class Board():

    '''
    Creating a board object

    Attrbutes:
    - size (int) = the dimension of a square n x n board
    - ships (array) = ships that occupy the board and its coordinates
    - num_fires (int) = numbers of times moves were made (hits or misses)
    - board_AI (array) = 2D array representing gameboard. 0 for ocean, -1 for miss, 100 for hit
    - actual_board (not seen by AI) (array) = 2D array representing board. 0 for ocean, 100 for actual ship locations

    Capabilites:
    - initalize_board = initalizes board and its values 
    - hit 
    - miss
    - Num_ships (# of squares occupied by ships currently)

    Behaviors:
    - missile() (coordinates : (int, int))  -- fires a missle at the coordinate
    - num_ships_left()                      -- give the numbers of ships still alive
    - score()                               -- updates current score
    - guess(x, y)                           -- guesses a coordinate
    '''
    
    # Define creation of a Board.
    def __init__(self, size):  # removed ships parameter since we start we zero anyway
        self.size = size   # set board to be size by size 
        self.ships = []    # Intitally no ships on the board, list of Ships
        self.num_fires = 0 # Initially the fire count is 0
        
        # list of ships, list of ship heads?
        # x = random.randint(0, size+1)
        # y = random.randint(0, size+1)
        # self.coordinates = (x, y) # random? -- coords of ship head

        # Board as seen by AI, should be instantiated as a 2d array of Os.
        # Updated as hits are made
        self.AIBoard = np.zeros((size, size))
        
        # Actual Board, should be instantiated as '_', other than where ships are located.
        # Instantiates as an empty board, then should add ships as provided.
        # Remains constant throughout entire game.
        # Changed value so it can by compatiable for the textual view
        self.HiddenBoard = np.full((size, size), '_')

    
    # Define the calculation of the score.
    # Does anyone know how to search through a 2d numpy arrray in faster time? I know the total time is 10 x 10 at max but.
    def score(self):
        # score = 0
        # For every row and column of the board, add the value to the total.
        # for row in Board.AIBoard: 
        #     for column in row:
        #         score =+ Board.AIBoard[row][column]
        # Return total score of every column in board.
        return np.sum(self.AIBoard)


    # How many misses have been made.
    def num_misses(self):
        misses = 0
        for row in range(0, len(self.AIBoard)):
            for column in range(0, len(self.AIBoard[0])):
                column = int(column)
                # If the value at each coordinate is equal to negative 1, add one to the count of misses.
                if self.AIBoard[row][column] == -1.0:
                    misses =+ 1
        return misses
    

    # Places a ship onto the board using the given coords (row, col)
    def place_ship(self, ship, row, col):
        # adds ship and its coordinates to list
        self.ships.append([ship, (row, col)]) 

        # marks its coors on the board (hidden board)
        # 'X' = part of a ship's positon on board - assuming it's more than 1 unit
        if (ship.dir == Direction.NORTH):
            for i in range(0, ship.size):
                self.HiddenBoard[row + i][col] = NONHIT_SHIP
        elif (ship.dir == Direction.SOUTH):
            for i in range(0, ship.size):
                self.HiddenBoard[row - i][col] = NONHIT_SHIP  
        elif (ship.dir == Direction.WEST):
            for i in range(0, ship.size):
                self.HiddenBoard[row][col + i] = NONHIT_SHIP  
        elif (ship.dir == Direction.EAST):
            for i in range(0, ship.size):
                self.HiddenBoard[row][col - i] = NONHIT_SHIP 


        
    def missile(self, coordinates):
        """" 
        launches a missle at the coordinate points

        Return -1 for a miss and return 100 for a hit
        AA missile launched at location already fired returns a miss
        """
        x = coordinates[0]
        y = coordinates[1]

        ship_coors = [ship[1] for ship in self.ships]

        # Add one to the total fires if (x, y) is a position of a ship and change its symbol on the hidden board
        # Add one to the total misses if (x, y) is a position of an empty tile and change its symbol on the hidden board
        if (x, y) in ship_coors:
            self.num_fires =+ 1
            self.HiddenBoard[x][y] = HIT_SHIP
        else:
            self.num_misses =+ 1
            self.HiddenBoard[x][y] = HIT_EMPTY

        # Make the reward at the hidden board visible on the AI board.
        # Maddie: I added .copy() to ensure they weren't editing each other but not sure if necessary.
        # Kene: commented out self.AIBoard[x][y] = self.HiddenBoard[x][y].copy() since arrays are of different types
        # self.AIBoard[x][y] = self.HiddenBoard[x][y].copy() 
        
        # checks if given coordinate is a miss or hit, given if (x, y) is in list of tuples (ships)
        # print('self.ships: ' + str(self.ships))
            

        if (x, y) in ship_coors:
            self.AIBoard[x][y] = 100  # set AIBoard[x][y] to 100 if a ship was hit (not sure if this is the right way to assign scores)
            ship_idx = ship_coors.index((x, y))
            self.ships[ship_idx][0].num_hit += 1  # update num_hit for the ship
            self.ships[ship_idx][0].num_left -= 1
            return 100
        else:
            self.AIBoard[x][y] = -1  # set AIBoard[x][y] to -1 if an empty tile was hit (not sure if this is the right way to assign scores)
            # self.ships[ship_idx].num_misses += 1  # update num_misses for the ship
            return -1


    def check_gameover(self): 
        """ Check gameover condition (all ships sunk). """
        total_life = sum([ship[0].num_left for ship in self.ships])

        if total_life == 0: 
            return True 
        else: 
            return False 
        

    # return number of ships left on board
    def num_ships_left(self):
        afloat_ships = 0

        for ship in self.ships:
            if ship[0].num_left > 0:
                afloat_ships += 1

        return afloat_ships

        

        
