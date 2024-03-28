from board import Board
from ship import Ship
from textualview import View

# SHIP: 
# Testing the initalization of a ship 
ship_starter = Ship("test boat", 1)

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

Ship.hit(ship_two)      

# print('Size is Correct?: ' + str(Ship.size(ship_two) == 1)) 
print('Correct ships left?: '+ str(ship_two.num_left == 1))     # true

Ship.hit(ship_two)  

print('Correct ships left?: '+ str(ship_two.num_left == 0))

print()


# BOARD:  
list_ships = [ship_starter, ship_two]
# Testing the initalization of the board
board_starter = Board(4, list_ships)

print("VALUES AT INITILALZATION")
print(board_starter)
print(board_starter.size == 4)
print(len(board_starter.ships) == 0)
print('Hidden Board: ')
print(board_starter.HiddenBoard)
print('AI Board: ')
print(board_starter.AIBoard)
# print('Does score begin at 0?: '+ str(board_starter.score() == 0.0))
# print('Does num of misses begin at 0?: '+ str(board_starter.num_misses() == 0.0))
print('0 ships on board at start?: '+ str(board_starter.num_ships_left() == 0.0))
print()

# Testing the gameplay methods of the board 
print("PLACING SHIPS ON BOARD")
ship1 = Ship("ship 1", 1)
ship2 = Ship("ship 2", 1)
ship3 = Ship("ship 3", 1)
ship4 = Ship("ship 4", 1)
ship5 = Ship("ship 5", 1)

board_starter.place_ship(ship1, 0, 1)
board_starter.place_ship(ship2, 1, 2)
board_starter.place_ship(ship3, 3, 3)
board_starter.place_ship(ship4, 1, 1)
board_starter.place_ship(ship5, 2, 1)

print('Five ships on board?: ' + str(len(board_starter.ships) == 5)) 
print('Hidden Board After Placing Ships: ')
print(board_starter.HiddenBoard)
print()

print('GAMEPLAY')
board_starter.missile((0, 1))
board_starter.missile((1, 1))
board_starter.missile((2, 1))
board_starter.missile((2, 2))

print('Number of Fires Increased?: ' + str(board_starter.num_fires == 4))  # True

print('Hidden Board After Hits: ')
print(board_starter.HiddenBoard)

print('AI Board After Hits: ')
print(board_starter.AIBoard)

print('Number of Ships Left?: ' + str(board_starter.num_ships_left() == 2))  # True
print('Game Over?: ' + str(board_starter.check_gameover()))  # False
print('Score of AI Board: ' + str(board_starter.score() == 299))  # True

board_starter.missile((3, 3))
board_starter.missile((1, 2))

print('Game Over?: ' + str(board_starter.check_gameover()))  # True

print('Hidden Board After Hits: ')
print(board_starter.HiddenBoard)

print('AI Board After Hits: ')
print(board_starter.AIBoard)

print('Score of AI Board: ' + str(board_starter.score() == 499))  # True

# view (may be unecessary ultimately
view = View()
print("VIEW:")
view.textual_view_draw(board_starter)