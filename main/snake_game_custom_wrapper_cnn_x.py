import math
import time

import gym
import numpy as np

from snake_game import SnakeGame

# MODEL_PATH_S = r"trained_models_cnn_finetuned/ppo_snake_final"
# # MODEL_PATH_L = r"trained_models_cnn_mask_finetuned_03/ppo_snake_26000000_steps" # Init success rate 0.44
# MODEL_PATH_L = r"trained_models_cnn_mask_finetuned_02/ppo_snake_37000000_steps" # Init success rate 0.46

class SnakeEnv(gym.Env):
    def __init__(self, silent_mode=True, seed=0, board_size=21, limit_step=True):
        super().__init__()
        self.game = SnakeGame(silent_mode=silent_mode, seed=seed, board_size=board_size)
        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        self.done = False
        self.reward_step_counter = 0

        self.reward_boost_factor = 3.0

        self.max_snake_length = self.game.board_size * self.game.board_size # Max length of snake is board_size*board_size, which is 21*21=441 for current setting.
        self.size_coefficient = 4.0 / self.max_snake_length # Keep the range of e^x in [-2, 2] 

        self.max_reward = self.game.board_size ** 2 - 140.0

    def reset(self):
        self.game.reset(warm_start=True)
        obs = self._generate_observation()

        self.done = False
        self.reward_step_counter = 0

        return obs
    
    def step(self, action):
        self.done, info = self.game.step(action) # info: "snake_length": int, "reward_steps": int, "snake_head_pos": np_array, "prev_snake_head_pos": np_array, "food_pos": np_array, "food_obtained": boolean
        obs = self._generate_observation()

        self.reward_step_counter += 1
        reward = 0

        # (441 + 140) / 2 = 290; 441 board size, 140 snake start length;
        # 441 - 140 = 301

        if info["food_obtained"]: # food eaten
            # Boost the reward based on snake length, the longer the snake, the bigger the reward.
            # reward = 1.0 / (1.0 + math.exp((self.max_snake_length/2 - info["snake_length"])*self.size_coefficient)) * 0.05
            # reward = 3 * math.pow(3, (info["snake_length"] - 140.0) / self.max_reward)
            if self.reward_step_counter <= 441: # fast, good
                reward = math.pow(3, (441.0 - self.reward_step_counter) / 441.0) * self.reward_boost_factor 
            else: # slow, bad
                reward = - math.pow(3, min(1, (self.reward_step_counter - 441.0) / 441.0))
            # Boost the reward based on num_steps between food eaten.
            self.reward_step_counter = 0 # Reset reward step counter
        
        elif self.done: # Bump into wall or exceed step limit, game over.
            # Shrink the penalty using snake length, the longer the snake, the smaller the penalty.
            # Max length of snake is board_size*board_size, which is 21*21=441 for current setting.
            # reward = - 1.0 / (1.0 + math.exp((info["snake_length"] - self.max_snake_length/2)*self.size_coefficient))
            if info["snake_length"] >= 290: # define as win
                reward = math.pow(300, (info["snake_length"] - 290.0) / 150) * self.reward_boost_factor
            else: # lose
                reward = - math.pow(300, (290.0 - info["snake_length"]) / 150)

            # reward = - math.pow(self.max_reward, (self.game.board_size**2 - info["snake_length"]) / self.max_reward)
        
        # max_score: 3*3*301 + 300*3 = 3609
        # low_score: -3*301-300 = -1203
        reward = reward * 0.003 # Scale reward to be in range [-3.6, 10.8]
        return obs, reward, self.done, info
    
    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall
        game_over = (
            (row, col) in snake_list
            or row < 0
            or row >= self.game.board_size
            or col < 0
            or col >= self.game.board_size
        )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)
        obs[tuple(np.transpose(self.game.snake))] = 200
        # Stack obs into 3 channels
        obs = np.stack((obs, obs, obs), axis=-1)
        
        # Set the snake head to green
        obs[tuple(self.game.snake[0])] = [0, 255, 0]

        # Set the food to red
        # obs[tuple(self.game.food)] = [0, 0, 255]

        for food in self.game.food_list:
            obs[tuple(food)] = [0, 0, 255]
        
        # Enlarge the observation to 84x84
        obs = np.repeat(np.repeat(obs, 4, axis=0), 4, axis=1)

        return obs

# Test the environment using random actions
NUM_EPISODES = 100
RENDER_DELAY = 0.001
from matplotlib import pyplot as plt

if __name__ == "__main__":
    env = SnakeEnv(silent_mode=False)
    
    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
