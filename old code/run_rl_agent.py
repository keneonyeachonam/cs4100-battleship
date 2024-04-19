import numpy as np
import gym 

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from gym.envs.registration import registry, make, spec

from rl_agent import RLAgent
from board import Board 
from battleship_gym_v7 import BattleshipEnvClass 


# CREATING OUR OWN GYM ENV
env_version = '1'

gym.envs.registration.register(id='BattleshipEnv-rl' + env_version,
         entry_point='battleship_gym_rl' + env_version + ':BattleshipEnvClass',
         max_episode_steps=1000,
         reward_threshold=2500000.0)

env = gym.make("BattleshipEnv-rl" + env_version)
model = PPO('MlpPolicy', env, verbose=0)
model = PPO.load("cs4100-battleship/ppo_battleship")


# ACTUALLU USING THE GYM ENV WE MADE
board = Board(10)

test_gym_env = BattleshipEnvClass()
agent = RLAgent(board, model, test_gym_env)
print(agent.play_until_completion())
