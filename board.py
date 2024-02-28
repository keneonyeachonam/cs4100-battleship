# Creating the board and its functions
import numpy as np

class Board():

    '''
    Creating a board object

    Attrbutes:
    - size (int) = the dimension of a square n x n board
    - ships (array) = ships that occupy the board
    - num_fires (int) = numbers of times moves were made (hits or misses)
    - board_AI (array) = 2D array representing gameboard. 0 for ocean, -1 for miss, 100 for hit
    - actual_board (not seen by AI) (array) = 2D array representing board. 0 for ocean, 100 for actual ship locations

    Capabilites:
    - initalize_board = initalizes board and its values 
    - hit 
    - miss
    - Score method
    - Num_misses
    - Num_attempts
    - Num_ships (# of squares occupied by ships currently)
    - valid_move 

    Behaviors:
    - missile() (coordinates : (int, int))  -- fires a missle at the coordinate
    - num_ships_left()                      -- give the numbers of ships still alive


    '''
    
    # Define creation of a Board.
    def __init__(self, size, ships):
        self.size = size
        self.ships = []    # Intitally no ships on the board, a list of tuples
        self.num_fires = 0 # Initially the fire count is 0
        
        # Board as seen by AI, should be instantiated as a 2d array of 0s.
        # Updated as hits are made
        self.AIBoard = np.zeros((size, size))
        
        # Actual Board, should be instantiated as -1s, other than where ships are located.
        # Instantiates as an empty board, then should add ships as provided.
        # Remains constant throughout entire game.
        self.HiddenBoard = np.full((size, size), -1)
        

    def guess(self, x, y):
        # Add one to the total fires.
        # This should update misses and make the appropriate changes.
        self.num_fires =+ 1

        # Make the reward at the hidden board visible on the AI board.
        # Maddie: I added .copy() to ensure they weren't editing each other but not sure if necessary or valid.
        # This needs to occur prior to seing if it is a hit, as the return will break from the funtcion.
        self.AIBoard[x][y] = self.HiddenBoard[x][y].copy()

        # checks if given coordinate is a miss or hit
 
        if (x, y) in self.ships:
            return 'hit'
        
        
        
    
    # Define the calculation of the score.
    # Does anyone know how to search through a 2d numpy arrray in faster time? I know the total time is 10 x 10 at max but.
    def score(self):
        score = 0
        # For every row and column of the board, add the value to the total.
        for row in self.AIBoard:
            for column in row:
                score =+ self.AIBoard[row][column]
        # Return total score of every column in board.
        return score
    
    # How many misses have been made.
    def num_misses(self):
        misses = 0
        for row in self.AIBoard:
            for column in row:
                # If the value at each coordinate is equal to negative 1, add one to the count of misses.
                if self.AIBoard[row][column] == -1:
                    misses =+ 1
        return misses
    
    # How many attempts have been made.
    def num_attempts(self):
        return self.num_fires
        

    #def place_ship(self, ship, row, col):
        # Places a ship onto the board 
        
        
    def missile(self, coordinates):
        """" launches a missle at the coordinate points

        Return -1 for a miss and return 100 for a hit
        AA missile launched at a location already fired returns a miss
        


        """


    # Check if the game should be finished.
    def check_gameover(self): 
        """ Check gameover condition (all ships sunk). """
        total_life = sum(ship.num_left for ship in self.ships)
        if total_life == 0: 
            return True 
        else: 
            return False 

    # Is the provided coordinate a valid move
    def valid_move(self, coordinate):
        # If this value has not already been guessed, return that this is a valid move.
        if self.AIBoard[coordinate[0]][coordinate[1]] == 0:
            return True
        # Otherwise, as there is a value already at this coordinate, it is not valid.
        return False
        

        