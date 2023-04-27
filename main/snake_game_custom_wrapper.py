import time

import gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, silent_mode=True):
        super().__init__()
        self.game = SnakeGame(silent_mode=silent_mode)
        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.game.board_size, self.game.board_size),
            dtype=np.float32
        ) # 0: empty, 0.25: snake body, 0.75: snake head, 1: food
        self.done = False
    
    def reset(self):
        self.game.reset()
        self.done = False
        return self._generate_observation()
    
    def step(self, action):
        food_obtained, self.done = self.game.step(action)
        obs = self._generate_observation()
        # reward = 0
        # if self.done:
        #     reward = -1
        # elif self.game.snake[0] == self.game.food:
        #     reward = 1
        # else:
        #     reward = -0.001 # Penalize the agent for every step it takes
        
        reward = 0
        if self.done:
            reward = -1
        elif food_obtained:
            reward = 1
        # else:
            # Use the distance between the snake head and the food as the reward
            # reward = np.exp(-np.linalg.norm(np.array(self.game.snake[0]) - np.array(self.game.food))/self.game.board_size)
            
        return obs, reward, self.done, {} # info is empty
    
    def render(self):
        self.game.render()

    # EMPTY: 0; SnakeBODY: 0.25; SnakeHEAD: 0.75; FOOD: 1;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.float32)
        obs[tuple(np.transpose(self.game.snake))] = 0.25
        obs[tuple(self.game.snake[0])] = 0.75
        obs[tuple(self.game.food)] = 1
        return obs

# Test the environment using random actions
NUM_EPISODES = 100
if __name__ == "__main__":
    env = SnakeEnv(silent_mode=False)
    sum_reward = 0
    for _ in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            sum_reward += reward
            if reward > 0:
                print(reward)
            env.render()
            time.sleep(0.01)
    
    env.close()
    print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
