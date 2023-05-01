import time
import random

from stable_baselines3 import PPO

from snake_game_custom_wrapper_cnn import SnakeEnv

MODEL_PATH_S = r"trained_models_cnn_finetuned/ppo_snake_final"
MODEL_PATH_L = r"trained_models_cnn_finetuned_04/ppo_snake_27500000_steps"
NUM_EPISODE = 10
RENDER_DELAY = 0.01

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

env = SnakeEnv(silent_mode=False, seed=seed, limit_step=False)

# Load the trained model
model_s = PPO.load(MODEL_PATH_S)
model_l = PPO.load(MODEL_PATH_L)

total_reward = 0
retry_limit = 10

for episode in range(NUM_EPISODE):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    num_step = 0
    snake_length = 3
    info = None
    print(f"=================== Episode {episode + 1} ==================")
    while not done:
        if info:
            snake_length = info["snake_length"]
        if snake_length < 43:
            action, _ = model_s.predict(obs)
        else:
            action, _ = model_l.predict(obs)

        # print(["UP", "LEFT", "RIGHT", "DOWN"][action])
        # Give the AI agent 10 retries if it collides with itself or the wall.
        i_retry = 0
        game_over = check_game_over(env, action)
        while game_over and i_retry < retry_limit:
            print(f"Retry {i_retry + 1}...")
            if snake_length < 43:
                action, _ = model_s.predict(obs)
            else:
                action, _ = model_l.predict(obs)
            game_over = check_game_over(env, action)
            i_retry += 1

        obs, reward, done, info = env.step(action)
        if info["food_obtained"]:
            print(f"Food obtained at step {num_step}, reward: {reward}")
        if done:
            final_direction = ["UP", "LEFT", "RIGHT", "DOWN"][action]
            print(f"Game over, penalty: {reward}")
            print(f"Final direction: {final_direction}")
            time.sleep(5)
        episode_reward += reward
        env.render()
        num_step += 1
        time.sleep(RENDER_DELAY)
    
    print(f"Episode {episode + 1}: Reward = {episode_reward}, Score = {env.game.score}, Total steps = {num_step}")
    total_reward += episode_reward

env.close()
print(f"=================== Summary ==================")
print(f"Average reward: {total_reward / NUM_EPISODE}")
