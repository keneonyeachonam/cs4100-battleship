from board import Board
import numpy as np

class RLAgent(): 

    def __init__(self, game_board, model, gym_env): 

        if game_board is None: 
            self.board = Board() 
        else: 
            self.board = game_board 

        self.model = model 

        self.env = gym_env  
        self.obs = self.env.reset() 
        self.env._overwrite_board(game_board)  


    def play_until_completion(self, debug=False): 
        """ 
        Plays game until complete. Returns score (torpedo count)
        """

        reward_list = list()
        episode_reward = 0 

        while True: 

            action, _states = self.model.predict(self.obs)  

            self.obs, reward, terminated, info = self.env.step(action)  
            episode_reward += reward 

            if terminated: 
                reward_list.append(episode_reward) 
                episode_reward = 0 
                break 

        return self.board.score(), episode_reward, reward_list   

