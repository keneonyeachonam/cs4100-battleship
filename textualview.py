class View():
    '''
    Rendering a basic textual view of a board's current state.
    Attributes: 
    board (Board): board to be rendered

    Capabilities:
    visualize_ship: renders a basic textual view of a ship
    textual_view_draw: renders a basic textual view of a board
    '''

# def visualize_ship(ship):
#     string = []

#     for tile in ship : #ship.body
#         if tile == 0:
#             string.append("O")
#         else:
#             string.append("X")
#     result = ''.join(string)

#     return result
        
    
def textual_view_draw():
    # should just iterate through every val in array
    # HIDDEN BOARD:
    # non-hit non-ship tiles represented by: _
    # HIT non-ship tiles represented by:     X
    # non-hit SHIP tiles represented by:     _
    # hit SHIP tiles represented by:         O

    # NON-HIDDEN BOARD:
    # non-hit non-ship tiles represented by: _
    # HIT non-ship tiles represented by:     !
    # non-hit SHIP tiles represented by:     O
    # hit SHIP tiles represented by:         X

    return None