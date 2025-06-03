import time
import random

from sb3_contrib import MaskablePPO

from snake_game_custom_wrapper_cnn import SnakeEnvCNN

NUM_EPISODES = 5000

RENDER = True
IS_SILENT = False
FRAME_DELAY = 0.05 # 0.02 fast, 0.05 slow
ROUND_DELAY = 1.5
PRINT = True

BOARD_SIZE = 6

MODEL_PATH = f"trained_models_cnn/{BOARD_SIZE}x{BOARD_SIZE}/ppo_final"

env = SnakeEnvCNN(board_size=BOARD_SIZE, enlarge_multiplier=84 / BOARD_SIZE, limit_step=True, is_render=RENDER, is_silent=IS_SILENT, cell_size=70*6/BOARD_SIZE, border_size=20)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_reward = 0
total_fruits = 0
min_fruits = int(1e9)
max_fruits = 0
wins = 0
total_win_steps = 0

start_time = time.time()

for episode in range(NUM_EPISODES):
    obs, info = env.reset(seed=random.randint(0, int(1e9)))
    episode_reward = 0
    terminated = False
    
    num_steps = 0
    info = None

    sum_step_reward = 0

    if PRINT:
        print(f"=================== Episode {episode + 1} ==================")

    while not terminated:
        action, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
        num_steps += 1
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            if info["snake_size"] == env.game.grid_size:
                wins += 1
                total_win_steps += num_steps
                if PRINT:
                    print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
            else:
                last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
                if PRINT:
                    print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

        elif info["food_obtained"]:
            if PRINT:
                print(f"Food obtained at step {num_steps:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            sum_step_reward = 0 

        else:
            sum_step_reward += reward
            
        episode_reward += reward
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    episode_fruits = env.game.fruits
    if episode_fruits < min_fruits:
        min_fruits = episode_fruits
    if episode_fruits > max_fruits:
        max_fruits = episode_fruits
    
    if PRINT:
        snake_size = info["snake_size"]
        print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Fruits: {episode_fruits}, Total Steps: {num_steps}, Snake Size: {snake_size}")
    total_reward += episode_reward
    total_fruits += env.game.fruits
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print("Model Path:", MODEL_PATH)
print("Trials:", NUM_EPISODES)
print(f"Average Fruits: {total_fruits / NUM_EPISODES}, Min Fruits: {min_fruits}, Max Fruits: {max_fruits}, Average reward: {total_reward / NUM_EPISODES}, Win Ratio: {wins / NUM_EPISODES}, Avg Moves to Win: {'no wins' if wins == 0 else total_win_steps/wins}")
print("Running Time (s):", time.time() - start_time)