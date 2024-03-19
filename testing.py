from board import Board
from ship import Ship

ship_starter = Ship("test boat", 1)

# # SHIP: 
# size(self)
# hit(self)
print(ship_starter)
print(Ship.size(ship_starter)) # size 1
print(Ship.hit(ship_starter)) # false
ship_starter.hit()

ship_two = Ship("bigger boat", 2)

print(ship_two.name)
print(Ship.hit(ship_two))

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


