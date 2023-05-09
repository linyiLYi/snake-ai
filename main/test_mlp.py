import time
import random

from stable_baselines3 import PPO

from snake_game_custom_wrapper_mlp import SnakeEnv

MODEL_PATH = r"trained_models/mlp_final"

NUM_EPISODE = 10

RENDER = True
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

if RENDER:
    env = SnakeEnv(seed=seed, silent_mode=False, limit_step=False)
else:
    env = SnakeEnv(seed=seed, silent_mode=True, limit_step=False)

# Load the trained model
model = PPO.load(MODEL_PATH)

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

    print(f"===================Episode {episode + 1}==================")
    while not done:
        action, _ = model.predict(obs)
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
