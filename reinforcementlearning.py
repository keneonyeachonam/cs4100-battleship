import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
import json 
import board
import game

# Might be irrelevant in terms of Mordor to Battleship conversion.
# Number of states, number of actions. 
nS = 100
nA = 100

# Do we have slip probability?
# slip_prob = 0.1

ships = []
ship_reward = 100

actions = ['up', 'down', 'left', 'right']  # Human readable labels for actions

p_0 = np.array([0 for _ in range(nS)])
# Marking as start state? We would need random or given through game.
p_0[12] = 1

P = np.zeros((nS,nS,nA), dtype=float)

# valid_neighbors no longer valid. Likely return list of unguessed squares?
def valid_neighbors(i,j):
    neighbors = {}
    if i>0:
        neighbors[0]=(i-1,j)
    if i<3:
        neighbors[1]=(i+1,j)
    if j>0:
        neighbors[2]=(i,j-1)
    if j<3:
        neighbors[3]=(i,j+1)
    return neighbors

for i in range(4):
    for j in range(4):
        if i==0 and j==2:
            continue            # outgoing probabilities from terminal states should be 0 in gymnasium
        if i==3 and j==1:
            continue            # outgoing probabilities from terminal states should be 0 in gymnasium

        neighbors = valid_neighbors(i,j)
        for a in range(nA):
            if a in neighbors:
                P[neighbors[a][0]*4+neighbors[a][1], i*4+j, a] = 1-slip_prob
                for b in neighbors:
                    if b != a:
                        P[neighbors[b][0]*4+neighbors[b][1], i*4+j, a] = slip_prob/float(len(neighbors.items())-1)

#################################################################
# REWARD MATRIX

# In this implementation, you only get the reward if you *intended* to get to 
# the target state with the corresponding action, but not through slipping.

# Doesn't really affect the implementation of your assignment questions below. 

#################################################################

# This might be state to state, no need for action.
R = np.zeros((nS, nS, nA))
# mark all as negative ones
R[:, :, :] = -1

# Set the value of reward for each ship to be that of a hit.
for ships in range(ships):
    # R[][][] = ship_reward
    # cannot set 100 action values by 100 by 100 times? How should rewards be fetermined?
    R[0][0][0] = ship_reward

# What id the matrix, would it need to change
env=gym.make('matrix_mdp/MatrixMDP-v0', p_0=p_0, p=P, r=R)

#################################################################
# Helper Functions
#################################################################

#reverse map observations in 0-15 to (i,j)
'''def reverse_map(observation):
    return observation//4, observation%4'''

#################################################################
# Q-Learning
#################################################################

'''

In this section, you will implement a function for Q-learning with epsilon-greedy exploration.
Refer to the written assignment for the update equation. Similar to MDPs, use the following code to take an action:

observation, reward, terminated, truncated, info = env.step(action)

Unlike MDPs, your action is now chosen by the epsilon-greedy policy. The action is chosen as follows:

With probability epsilon, choose a random legal action.
With probability (1 - epsilon), choose the action that maximizes the Q-value (based on the last estimate). 
In case of ties, choose the action with the smallest index.

In case the chosen action is not a legal move, generate a random legal action.

The episode terminates when the agent reaches one of two terminal states. 

The Q-table is initialized to all zeros. The value of eta is unique for every (s,a) pair, and
should be updated as 1/(1 + number of updates to Q_opt(s,a)) inside the loop. 

The number of updates to Q_opt(s,a) should be stored in a matrix of shape (nS, nA) initialized to zeros, 
and updated such that num_updates[s,a] gives you the number of times Q_opt(s,a) has been updated.
You can then calculate eta using the formula above.

The value of epsilon should be decayed to (0.9999 * epsilon) at the end of each episode.

After 10, 100, 1000 and 10000 episodes, plot a heatmap of V_opt(s) for all states s. Complete and use the plot_heatmaps() function. 
The heatmap should be a 4x4 grid, corresponding to our map of Mordor. Please use plt.savefig() to save the plot, and do not use plt.show().
Add each heatmap (clearly labeled) to your answer to Q9 in the written submission.

'''

def q_learning(num_episodes, checkpoints):
    """
    Q-learning algorithm.
    Parameters:
    - num_episodes (int): Number of Q-value episodes to perform.
    - checkpoints (list): List of episode numbers at which to record the optimal value function..
    Returns:
    - Q (numpy array): Q-values of shape (nS, nA) after all episodes.
    - optimal_policy (numpy array): Optimal policy, np array of shape (nS,), ordered by state index.
    - V_opt_checkpoint_values (list of numpy arrays): A list of optimal value function arrays at specified episode numbers.
      The saved values at each checkpoint should be of shape (nS,).
    """
    # Set Q to be the reward at each state
    Q = np.zeros((nS, nA))  
    num_updates = np.zeros((nS, nA))  

    # These should be good values. Can alter if needed. 
    gamma = 0.9  
    epsilon = 0.9 

    # Do we need checkpoint values?
    V_opt_checkpoint_values = [] 

    for episode in tqdm(range(num_episodes)):
        terminated = False
        init_observation = env.reset()
        # debugged type issues with env.reset() using gen AI
        # I do not know what this does.
        if isinstance(init_observation, int):
            observation = init_observation
        else:
            observation = init_observation[0]
        
        while not terminated:
            current_coords = reverse_map(observation)
            new_cell = observation

            random = True
            new_cell_coords = (-1, -1)
            action = -1
        
            # If less than epsilon, chose some random action
            # This should ensure that it is not already searched. 
            if np.random.rand() < epsilon:
                action = np.random.randint(0, nA)
            # Otherwise choose best action
            # Also needs to be ensured that its not already guessed.
            else:
                action = np.argmax(Q[observation])
                random = False  

            # While the new destination is not valid, generate new action
            while new_cell_coords not in valid_neighbors(*current_coords).values():
                if random:
                    action = np.random.randint(0, nA)
                else:
                    random = True

                match action:
                    case 0:
                        new_cell = observation - 4
                    case 1:
                        new_cell = observation + 4
                    case 2:
                        new_cell = observation - 1
                    case 3:
                        new_cell = observation + 1

                new_cell_coords = reverse_map(new_cell)
            
            prev_observation = observation  
            observation, reward, terminated, truncated, info = env.step(action)

            # debugged calculations with use of gen AI
            best_reward = np.max(Q[observation])  
            eta = 1 / (1 + num_updates[prev_observation, action]) 
            Q[prev_observation, action] += eta * (reward + gamma * best_reward - Q[prev_observation, action])
            num_updates[prev_observation, action] += 1 

        epsilon *= 0.9999

        if episode + 1 in checkpoints:
            V_opt = np.max(Q, axis=1)
            V_opt_checkpoint_values.append(V_opt) 

    optimal_policy = np.argmax(Q, axis=1) 

    return Q, optimal_policy, V_opt_checkpoint_values

# Replaced 4s from mordor with size.
# May need to be 10, or the size as generated.
size = 10

def plot_heatmaps(V_opt, filename):
    V_opt_reshaped = np.reshape(V_opt, (size, size)) 
    fig, ax = plt.subplots()
    heatmap = ax.imshow(V_opt_reshaped, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    
    for i in range(size):
        for j in range(size):
            text = ax.text(j, i, round(V_opt_reshaped[i, j], 2),
                           ha="center", va="center", color="w")

    plt.savefig(filename)
    plt.close()
    """
    Plots a 4x4 heatmap of the optimal value function, with state positions 
    corresponding to cells in the map of Mordor, with the given filename.

    Do not use plt.show().

    Parameters:
    V_opt (numpy array): A numpy array of shape (nS,) representing the optimal value function.
    filename (str): The filename to save the plot to. 

    Returns:
    None
    """

def main():

    Q, optimal_policy, V_opt_checkpoint_values = q_learning(10000, checkpoints=[10,100,1000,10000])

    plot_heatmaps(V_opt_checkpoint_values[0], "checkpoint_10.png")
    plot_heatmaps(V_opt_checkpoint_values[1], "checkpoint_100.png")
    plot_heatmaps(V_opt_checkpoint_values[2], "checkpoint_1000.png")
    plot_heatmaps(V_opt_checkpoint_values[3], "checkpoint_10000.png")


    #######################################
    # DO NOT CHANGE THE FOLLOWING - AUTOGRADER JSON DUMP
    #######################################

# Do not think the autograder is needed.
'''
    answer = {
        "V_s_10": V_opt_checkpoint_values[0].tolist(),
        "V_s_100": V_opt_checkpoint_values[1].tolist(),
        "V_s_1000": V_opt_checkpoint_values[2].tolist(),
        "V_s_10000": V_opt_checkpoint_values[3].tolist(),
        "optimal_policy": optimal_policy.tolist(),
        "mordor_q_table": Q.tolist(),
    }
    
    with open("answers_mordor.json", "w") as outfile:
        json.dump(answer, outfile)


if __name__ == "__main__":
    main()'''