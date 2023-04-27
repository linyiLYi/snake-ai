import time
import pygame
from stable_baselines3 import PPO
from snake_game_custom_wrapper import SnakeEnv

from snake_game import SnakeGame

MODEL_NAME = r"ppo_snake"
def test_trained_model(model_path, num_episodes=10, render_delay=0.1):
    env = SnakeEnv()
    pygame.init()
    pygame.display.set_caption("Snake Game")
    env.game.screen = pygame.display.set_mode((env.game.display_width, env.game.display_height))
    env.game.font = pygame.font.Font(None, 36)
    # Load the trained model
    model = PPO.load(model_path)
    
    total_reward = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(render_delay)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
        total_reward += episode_reward
    
    env.close()
    print(f"Average reward: {total_reward / num_episodes}")

if __name__ == "__main__":
    test_trained_model(MODEL_NAME, num_episodes=10, render_delay=0.1)
