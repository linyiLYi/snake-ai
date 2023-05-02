import math
import time

import gym
import numpy as np

from snake_game import SnakeGame

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

        # Constants
        if limit_step:
            self.step_limit = 3 * self.game.board_size * self.game.board_size # Experimental value: 3*board_size^2 steps should be more than enough to get the food, the snake should not keep looping around
        else:
            self.step_limit = 1e9 # No step limit
        self.max_snake_length = self.game.board_size * self.game.board_size # Max length of snake is board_size*board_size, which is 21*21=441 for current setting.
        self.size_coefficient = 4.0 / self.max_snake_length # Keep the range of e^x in [-2, 2] 

    def reset(self):
        self.game.reset()
        obs = self._generate_observation()
        
        self.done = False
        self.reward_step_counter = 0

        return obs
    
    def step(self, action):
        self.done, info = self.game.step(action) # info: "snake_length": int, "reward_steps": int, "snake_head_pos": np_array, "prev_snake_head_pos": np_array, "food_pos": np_array, "food_obtained": boolean
        obs = self._generate_observation()

        self.reward_step_counter += 1

        if self.reward_step_counter > self.step_limit:
            self.done = True # Set game to lose if agent go beyond step limit
            self.reward_step_counter = 0 

        reward = 0
        if info["food_obtained"]: # food eaten
            # Boost the reward based on snake length, the longer the snake, the bigger the reward.
            # Sigmoid reward with 1.0 as constant boost.
            reward = 1.0 + 1.0 / (1.0 + math.exp((self.max_snake_length/2 - info["snake_length"])*self.size_coefficient))
            self.reward_step_counter = 0 # Reset reward step counter
        
        elif self.done: # Bump into wall or exceed step limit, game over.
            # Shrink the penalty using snake length, the longer the snake, the smaller the penalty.
            # Max length of snake is board_size*board_size, which is 21*21=441 for current setting.
            reward = - 1.0 / (1.0 + math.exp((info["snake_length"] - self.max_snake_length/2)*self.size_coefficient))
            
            # Increase failure penalty in finetuning #03
            # reward = reward * 30

            # Further increase failure penalty in finetuning #04
            # reward = reward * 60

            # Further increase failure penalty in cnn_mask_finetuned #01
            # reward = reward * 120

            # Further increase failure penalty in cnn_mask_finetuned #01
            reward = reward * 220

        # else:
        #     # Reward/punish the agent based on whether it is heading towards the food or not.
        #     # Use a shrinking coefficient to make it a small incentive not competing with the win/lose reward.
        #     if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
        #         reward = 0.01 # max_cumulated_reward = reward * board_size * 2  
        #     else:
        #         reward = -0.01

        # Reward normalization in finetuning #03, 04
        reward = reward * 0.1

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
NUM_EPISODES = 10
RENDER_DELAY = 0.001
from matplotlib import pyplot as plt

if __name__ == "__main__":
    env = SnakeEnv(silent_mode=False)
    sum_reward = 0

    # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    for _ in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        i = 0
        while not done:
            plt.imshow(obs, interpolation='nearest')
            plt.show()
            action = env.action_space.sample()
            # action = action_list[i]
            i = (i + 1) % len(action_list)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
            if np.absolute(reward) > 0.001:
                print(reward)
            env.render()
            
            time.sleep(RENDER_DELAY)
        # print(info["snake_length"])
        # print(info["food_pos"])
        # print(obs)
        print("sum_reward: %f" % sum_reward)
        print("episode done")
        # time.sleep(100)
    
    env.close()
    print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
