import numpy as np
import gym
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

env = gym.make()

'''
NOTES FROM MEETING - 040324
- use a neural network (CNN)
   - start with 2x2 kernel with the 5x5 board, then do 2x2 with the resulting 4x4 board (may need only one!)
   - flatten result + each input connects with 25 outputs in our CNN
   - each output corresponds to a cell in board
   - use hw 3 CNN as skeleton code (also tutorials online)
   - should be less than 10 lines of code
   - layers: Convultion layer ->  nn.Linear (9, 25) (not working? try number between 9 and 25) -> Softmax
   - keep track of total loss (L = -log(P(a | s) * R) and do backprogation to get min loss
- flatten board to a (25, 1) array 
- use softmax as last layer (pick max prob, if invalid pick next biggest probabilty)
'''

