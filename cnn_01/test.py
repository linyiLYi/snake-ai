import time
import random

from stable_baselines3 import PPO

from snake_game_custom_wrapper import SnakeEnv

MODEL_PATH = r"trained_models/ppo_snake_50000000_steps"
NUM_EPISODE = 10
RENDER_DELAY = 0.01

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")
env = SnakeEnv(silent_mode=False, seed=seed)

# Load the trained model
model = PPO.load(MODEL_PATH)

total_reward = 0

for episode in range(NUM_EPISODE):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    num_step = 0
    print(f"===================Episode {episode + 1}==================")
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if info["food_obtained"]:
            print(f"Food obtained at step {num_step}, reward: {reward}")
        if done:
            print(f"Game over, penalty: {reward}")
        episode_reward += reward
        env.render()
        num_step += 1
        time.sleep(RENDER_DELAY)
    
    print(f"Episode {episode + 1}: Reward = {episode_reward}, Score = {env.game.score}, Total steps = {num_step}")
    total_reward += episode_reward

env.close()
print(f"Average reward: {total_reward / NUM_EPISODE}")