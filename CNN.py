import numpy as np

def CNN():
    array_size = 3
    output = np.zeros(25)

    # Convolution Layer
    # Create selection of 3 x 3 arrays to feed into CNN
    # Slide over board until all have been covered
    arrays = []
    for x in range(array_size):
        for y in range(array_size):
            addition = [(x, y), (x + 1, y), (x + 2, y), (x, y + 1), (x + 1, y + 1), (x + 2, y + 1), (x, y + 2), (x + 1, y + 2), (x + 2, y + 2)]
            arrays.append(addition)

    # Outputs this: 
    '''
    [[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
    [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)],
    [(0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4)],
    [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)], 
    [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)], 
    [(1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4)], 
    [(2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2)], 
    [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (4, 3)], 
    [(2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (4, 3), (2, 4), (3, 4), (4, 4)]]
    '''
    
    # ReLu layer

    # Pooling Layer

    return output


# Select the highest valid coordinate to guess
def selection():
    # The 25 size vector
    output = CNN()

    # The coordinate at which the highest predicted value
    val = binarysearch(max(output), output)
    # Put into coordinate format
    ans = coord_format(val)

    # Ensure valid move at ans, if not, then find next highest valid solution.
    while ans:
        # select next highest value.
        val = binarysearch(max(output), output)
        ans = coord_format(val)

    # Return the coordinate selected
    return ans


def coord_format(val):
    return (val % 5, np.floor_divide(val, 5))