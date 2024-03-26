from board import Board
from ship import Ship

ship_starter = Ship("test boat", 1)

# SHIP: 
# Testing the initalization of a ship 
print(ship_starter)
print(Ship.size(ship_starter) == 1)         # true
print(Ship.hit(ship_starter))               # false

ship_starter.hit()

print(Ship.num_left(ship_starter) == 0)     # true
print(ship_starter.isSunk == True)          # true
print(ship_starter.num_ships_left == 0)          # true


ship_two = Ship("bigger boat", 2)
print(ship_two.name) 
print(Ship.hit(ship_two) == 1)          # true
print(Ship.size(ship_two) == 1)         # true
print(Ship.num_left(ship_two) == 1)     # true

# print(board_starter)

# BOARD: 
# score
# num_misses
# place_ship
# missile(self, coordinates)
# check_gameover(self)
# num_ships_left()
board_starter = Board(4, ship_starter)

print(board_starter)
print(board_starter.score())


