import time
import random

import numpy as np
from sb3_contrib import MaskablePPO
# import matplotlib.pyplot as plt # For checking raw observation.

from snake_game_custom_wrapper_v2 import SnakeEnv

MODEL_PATH = r"trained_models_new_snake_and_reward/ppo_snake_35000000_steps"

NUM_EPISODE = 10

RENDER = False
FRAME_DELAY = 0.001 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

# Check if the action ends the game
def check_game_over(env, action):
    snake_list = env.game.snake
    row, col = snake_list[0]
    current_direction = env.game.direction
    if action == 0: # UP
        if current_direction != "DOWN":
            row -= 1
        else:
            row += 1

    elif action == 1: # LEFT
        if current_direction != "RIGHT":
            col -= 1
        else:
            col += 1

    elif action == 2: # RIGHT
        if current_direction != "LEFT":
            col += 1
        else:
            col -= 1

    elif action == 3: # DOWN
        if current_direction != "UP":
            row += 1
        else:
            row -= 1

    # Check if snake collided with itself or the wall
    game_over = (
        (row, col) in snake_list
        or row < 0
        or row >= env.game.board_size
        or col < 0
        or col >= env.game.board_size
    )

    return game_over

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")


if RENDER:
    env = SnakeEnv(seed=seed, silent_mode=False)
else:
    env = SnakeEnv(seed=seed, silent_mode=True)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0

for episode in range(NUM_EPISODE):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    num_step = 0
    info = None

    sum_step_reward = 0

    retry_limit = 9
    print(f"=================== Episode {episode + 1} ==================")
    while not done:
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        num_step += 1

        # Give the AI agent 10 retries if it collides with itself or the wall.
        i_retry = 0
        game_over = check_game_over(env, action)

        # Testing the mask.
        # print(env.get_action_mask())
        # time.sleep(1)

        # if game_over:
        #     print(f"Action Choice: {['UP', 'LEFT', 'RIGHT', 'DOWN'][action]}")
        #     print(env.get_action_mask())
        #     print(f"Current direction: {env.game.direction}")
        #     time.sleep(5000)

        # while game_over and i_retry < retry_limit:
        #     action, _ = model.predict(obs, action_masks=env.get_action_mask())
        #     print(f"Retry {i_retry + 1}: {['UP', 'LEFT', 'RIGHT', 'DOWN'][action]}")
        #     game_over = check_game_over(env, action)
        #     i_retry += 1
        
        # Check observation.
        # plt.imshow(obs, interpolation='nearest')
        # plt.show()

        obs, reward, done, info = env.step(action)
        
        if info["food_obtained"]:
            # print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f} ({info['food_reward_on_speed']:.4f}+{info['food_reward_on_size']:.4f}). Step Reward: {sum_step_reward:.4f}")
            # print(info["reward_step_counter"]) # Debug
            sum_step_reward = 0 # Debug
        elif done:
            final_direction = ["UP", "LEFT", "RIGHT", "DOWN"][action]
            print(f"Gameover Penalty: {reward:.4f}. Final direction: {final_direction}")
        else:
            sum_step_reward += reward
            # print(info["step_reward"], info["snake_size"]) # Debug
            
        episode_reward += reward
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    
    snake_size = info["snake_size"] + 1
    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}")
    total_reward += episode_reward
    total_score += env.game.score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")
