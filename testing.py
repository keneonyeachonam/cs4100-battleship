from board import Board
from ship import Ship

ship_starter = Ship("test boat", 1)

# SHIP: 
# Testing the initalization of a ship 
print("First ship to test: " + str(ship_starter.name))
print("Correct size?: " + str(Ship.size(ship_starter) == 1))      # true
print('Ship sunk yet?: ' + str(Ship.hit(ship_starter)))           # true
print('No more ships?: ' + str(ship_starter.num_left == 0))            # true
print('Check isSunk field: ' + str(ship_starter.isSunk == True))    # true
# print(ship_starter.num_ships_left == 0)     # true
print()


ship_two = Ship("bigger boat", 2)
print('Ship Name: ' + ship_two.name) 
print('Orginial Size is Correct?: ' + str(Ship.size(ship_two) == 2))         # true
Ship.hit(ship_two) == 1        # true
print('Size is Correct?: ' + str(Ship.size(ship_two) == 1)) 
print('Correct ships left?: '+ str(ship_two.num_left== 1))     # true
print()

# print(board_starter)

# BOARD: 
# score
# num_misses
# place_ship
# missile(self, coordinates)
# check_gameover(self)
# num_ships_left()
board_starter = Board(4) # ship_starter
ship1 = Ship("ship 1", 1)

print("Values at Intialization")
print(board_starter)
print(board_starter.size == 4)
print(len(board_starter.ships) == 0)
print('Hidden Board: ')
print(board_starter.HiddenBoard)
print('AI Board: ')
print(board_starter.AIBoard)
print('Does score begin at 0?: '+ str(board_starter.score() == 0.0))
print('Does num of misses begin at 0?: '+ str(board_starter.num_misses() == 0.0))
print('0 ships on board at start?: '+ str(board_starter.num_ships_left() == 0.0))
print()

print("Placing Ships on Board")
board_starter.place_ship(ship1, 0, 1)

print('Update correct? ' + str(len(board_starter.ships) == 1)) 


