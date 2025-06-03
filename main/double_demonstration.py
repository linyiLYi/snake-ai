from snake_game import SnakeGame
from snake_game_custom_wrapper_cnn import SnakeEnvCNN
from snake_game_custom_wrapper_mlp import SnakeEnvMLP
from sb3_contrib import MaskablePPO

import random
import pygame
import sys
import time

if __name__ == "__main__":
    board_size = 6
    is_silent = False

    seed = random.randint(0, int(1e9))
    print("Seed:", seed)

    env = SnakeEnvCNN(seed=seed, board_size=board_size, is_render=True, is_silent=is_silent, enlarge_multiplier=84/board_size, cell_size=70*6/board_size, border_size=20)
    obs, info = env.reset()
    env_steps = 0

    game = SnakeGame(seed=seed, board_size=board_size, is_render=True, is_silent=is_silent, cell_size=70*6/board_size, border_size=20)
    game.done = False
    game_steps = 0

    game_action = -1

    screen = game.screen = env.game.screen = pygame.display.set_mode((game.display_width * 2, game.display_height))

    x_offset = env.game.display_width
    game.render(x_offset=x_offset)
    env.render()

    MODEL_PATH = f"trained_models_cnn/{board_size}x{board_size}/ppo_final"
    model = MaskablePPO.load(MODEL_PATH)

    should_update = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_LEFT 
                    or event.key == pygame.K_RIGHT
                    or event.key == pygame.K_UP
                    or event.key == pygame.K_DOWN):
                    should_update = True
                if event.key == pygame.K_UP:
                    game_action = 0
                elif event.key == pygame.K_DOWN:
                    game_action = 3
                elif event.key == pygame.K_LEFT:
                    game_action = 1
                elif event.key == pygame.K_RIGHT:
                    game_action = 2

            if event.type == pygame.QUIT:
                game.close()
                sys.exit()

        if should_update:    
            if not game.done:
                game_steps += 1
                game.done, game_info = game.step(game_action)
            game.render(x_offset=x_offset)
            
            if not env.terminated:
                env_steps += 1
                env_action, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
                obs, _, terminated, truncated, env_info = env.step(env_action)
            env.render()
            
            should_update = False
        if game.done:
            time.sleep(1.5)
            new_seed = random.randint(0, 1e9)
            print("Seed:", new_seed)
            game.seed = env.game.seed = new_seed
            game.reset()
            game.done = False
            env.reset()

            game.render(x_offset=x_offset)
            env.render()