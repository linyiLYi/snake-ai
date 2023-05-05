import time
import random

from sb3_contrib import MaskablePPO

from snake_game_custom_wrapper_cnn import SnakeEnv

MODEL_PATH_S = r"trained_models/cnn_mask_og_01_150000000_steps"  # Orignial training. ("og" for original)
MODEL_PATH_L = r"trained_models/cnn_mask_ff_02_53500000_steps"   # With num_food = 40. ("ff" for forty food)
MODEL_PATH_X = r"trained_models/cnn_mask_ws_04_8500000_steps"    # With warm start length of 140. ("ws" for warm start)

RENDER = False
NUM_EPISODE = 3
FRAME_DELAY = 0.01 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

if RENDER:
    env = SnakeEnv(seed=seed, silent_mode=False, limit_step=False)
else:
    env = SnakeEnv(seed=seed, silent_mode=True, limit_step=False)

# Load the trained model
model_s = MaskablePPO.load(MODEL_PATH_S)
model_l = MaskablePPO.load(MODEL_PATH_L)
model_x = MaskablePPO.load(MODEL_PATH_X)

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

    print(f"=================== Episode {episode + 1} ==================")
    while not done:
        if info:
            snake_size = info["snake_length"]
        else:
            snake_size = 3
        
        if snake_size < 40:
            action, _ = model_s.predict(obs, action_masks=env.get_action_mask())
        elif snake_size < 140:
            action, _ = model_l.predict(obs, action_masks=env.get_action_mask())
        else:
            action, _ = model_x.predict(obs, action_masks=env.get_action_mask())

        obs, reward, done, info = env.step(action)
        num_step += 1
        if info["food_obtained"]:
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}.")
        elif done:
            final_direction = ["UP", "LEFT", "RIGHT", "DOWN"][action]
            print(f"Gameover Penalty: {reward:.4f}. Final Direction: {final_direction}")
            
        episode_reward += reward
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    
    snake_size = info["snake_length"] + 1
    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}")
    total_reward += episode_reward
    total_score += env.game.score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")
