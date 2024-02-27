# Creating the board and its functions

class Board():

    '''
    Creating a board object

    Attrbutes:
    - size (int) = the dimension of a square n x n board
    - ships (array) = ships that occupy the board
    - num_fires (int) = numbers of times moves were made (hits or misses)
    - board_AI (array) = 2D array representing gameboard. 0 for ocean, -1 for miss, 100 for hit
    - actual_board (not seen by AI) (array) = 2D array representing board. 0 for ocean, 100 for actual ship locations

    Capaabilites:
    - initalize_board = initalizes board and its values 
    - hit 
    - miss
    - Score method
    - Num_misses
    - Num_attempts
    - Num_ships (# of squares occupied by ships currently)


    '''
    
    # Define creation of a Board.
    def __init__(self, size):
        self.size = size
        