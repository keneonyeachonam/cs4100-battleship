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
    - Num_ships (# of squares occupied by ships currently)

    Behaviors:
    - missile() (coordinates : (int, int))  -- fires a missle at the coordinate
    - num_ships_left()                      -- give the numbers of ships still alive
    - score()                               -- updates current score
    - guess(x, y)                           -- guesses a coordinate
    '''
    
    # Define creation of a Board.
    def __init__(self, size, ships):
        self.size = size
        self.ships = []    # Intitally no ships on the board, list of Ships
        self.num_fires = 0 # Initially the fire count is 0
        
        # list of ships, list of ship heads?
        # x = random.randint(0, size+1)
        # y = random.randint(0, size+1)
        # self.coordinates = (x, y) # random? -- coords of ship head

        # Board as seen by AI, should be instantiated as a 2d array of 0s.
        # Updated as hits are made
        self.AIBoard = np.zeros((size, size))
        
        # Actual Board, should be instantiated as -1s, other than where ships are located.
        # Instantiates as an empty board, then should add ships as provided.
        # Remains constant throughout entire game.
        self.HiddenBoard = np.full((size, size), -1)

    
    # Define the calculation of the score.
    # Does anyone know how to search through a 2d numpy arrray in faster time? I know the total time is 10 x 10 at max but.
    def score():
        score = 0
        # For every row and column of the board, add the value to the total.
        for row in Board.AIBoard: 
            for column in row:
                score =+ Board.AIBoard[row][column]
        # Return total score of every column in board.
        return score


    # How many misses have been made.
    def num_misses(self):
        misses = 0
        for row in Board.AIBoard:
            for column in row:
                # If the value at each coordinate is equal to negative 1, add one to the count of misses.
                if Board.AIBoard[row][column] == -1:
                    misses =+ 1
        return misses
    


    def place_ship(self, ship, row, col):
        # Places a ship onto the board using the given coords (row, col)
        self.ships.append(ship)            # places ship in list
        self.HiddenBoard[row][col] = 'X'   # marks its coors on the board (hidden board)


        
    def missile(self, coordinates):
        """" 
        launches a missle at the coordinate points

        Return -1 for a miss and return 100 for a hit
        AA missile launched at  location already fired returns a miss
        """
        x = coordinates[0]
        y = coordinates[1]

        # Add one to the total fires.
        # This should update misses and make the appropriate changes.
        Board.num_fires =+ 1

        # Make the reward at the hidden board visible on the AI board.
        # Maddie: I added .copy() to ensure they weren't editing each other but not sure if necessary.
        Board.AIBoard[x][y] = Board.HiddenBoard[x][y].copy() 
        
        # checks if given coordinate is a miss or hit, given if (x, y) is in list of tuples (ships)
        ship_coors = [Board.ships[i].coordinates for i in Board.ships]
        if (x, y) in ship_coors:
            # update num_hit for the ship
            ship_idx = ship_coors.index((x, y))
            Board.ships[ship_idx].num_hit += 1
            Board.ships[ship_idx].num_left -= 1
            return 100
        else:
            # update num_misses for the ship
            Board.ships[ship_idx].num_misses += 1
            return -1


    def check_gameover(self): 
        """ Check gameover condition (all ships sunk). """
        total_life = sum([ship.num_left for ship in self.ships])

        if total_life == 0: 
            return True 
        else: 
            return False 
        

    # return number of ships left on board
    def num_ships_left():
        afloat_ships = 0

        for ship in Board.ships:
            if ship.num_left > 0:
                afloat_ships += 1

        return afloat_ships

        

        
