class View():
    '''
    Rendering a basic textual view of a board's current state.
    Attributes: 
    board (Board): board to be rendered

    Capabilities:
    visualize_ship: renders a basic textual view of a ship
    textual_view_draw: renders a basic textual view of a board
    '''

    def __init__(self, name):
        self.name = name 

# def visualize_ship(ship):
#     string = []

#     for tile in ship : #ship.body
#         if tile == 0:
#             string.append("O")
#         else:
#             string.append("X")
#     result = ''.join(string)

#     return result


    def textual_view_draw(self, board):
        ai_string = []
        hidden_string = []

        # ai board
        for row in board.AIBoard:
            for tile in row:
                if tile == 0: # non-ship tile
                    ai_string.append("_")
                elif tile == -1: # non-ship tile has been hit
                    ai_string.append("X")
                elif tile == 100: # ship tile has been hit
                    ai_string.append("O")

        # hidden board
        for row in board.HiddenBoard:
            for tile in row:
                if tile == 0:
                    ai_string.append("_") 
                elif tile == 100:
                    ai_string.append("O")

        ai_result = ''.join(ai_string)
        hidden_result = ''.join(hidden_string)

        return ai_result + "\n" + hidden_result