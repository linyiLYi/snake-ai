import time
import random

from sb3_contrib import MaskablePPO
# import numpy as np # For debugging.
# import matplotlib.pyplot as plt # For checking raw observation.

from snake_game_custom_wrapper_mlp import SnakeEnv

MODEL_PATH = r"trained_models_mlp_05_limit_linear_reward/ppo_snake_5000000_steps"
# MODEL_PATH = r"trained_models_mlp_03_no_limit/ppo_snake_8000000_steps" # Circling

NUM_EPISODE = 10

RENDER = True
FRAME_DELAY = 0.001 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")


if RENDER:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=False)
else:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=True)

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
        
        prev_mask = env.get_action_mask()
        # if np.sum(prev_mask) <= 1:
        #     print(prev_mask)
        #     time.sleep(5)
        prev_direction = env.game.direction

        num_step += 1
        
        # Check observation.
        # plt.imshow(obs, interpolation='nearest')
        # plt.show()

        obs, reward, done, info = env.step(action)
        
        if done:
            last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
            print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")
            
            # print(f"Previous direction: {prev_direction}")
            # print(f"Final direction: {env.game.direction}")
            # print(f"Prev mask: {prev_mask}")
            # print(f"Current mask: {env.get_action_mask()}")
            # time.sleep(6000)
        
        elif info["food_obtained"]:
            # print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            # print(info["reward_step_counter"]) # Debug
            sum_step_reward = 0 # Debug

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
