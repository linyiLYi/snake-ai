import time
import random

from snake_game_custom_wrapper_cnn import SnakeEnv

FRAME_DELAY = 0.01 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

BOARD_SIZE = 12

def generate_hamiltonian_cycle(board_size):
    path = [(0, c) for c in range(board_size)]

    for i in range(1, board_size):
        if i % 2 == 0:
            for j in range(1, board_size):
                path.append((i, j))
        else:
            for j in range(board_size - 1, 0, -1):
                path.append((i, j))

    for r in range(board_size - 1, 0, -1):
        path.append((r, 0))
    
    return path

def find_next_action(snake_head, next_position):
    row_diff = next_position[0] - snake_head[0]
    col_diff = next_position[1] - snake_head[1]

    if row_diff == 1 and col_diff == 0:
        return 3  # DOWN
    elif row_diff == -1 and col_diff == 0:
        return 0  # UP
    elif row_diff == 0 and col_diff == 1:
        return 2  # RIGHT
    elif row_diff == 0 and col_diff == -1:
        return 1  # LEFT
    else:
        return -1

def main():
    seed = random.randint(0, 1e9)
    print(f"Using seed = {seed} for testing.")

    env = SnakeEnv(silent_mode=False, seed=seed, board_size=BOARD_SIZE)

    cycle = generate_hamiltonian_cycle(env.game.board_size)
    cycle_len = len(cycle)
    current_index = 0

    num_step = 0
    done = False

    while not done:
        action = env.action_space.sample()
        snake_head = env.game.snake[0]
        current_index = (current_index + 1) % cycle_len

        while cycle[current_index] != snake_head:
            current_index = (current_index + 1) % cycle_len

        next_position = cycle[(current_index + 1) % cycle_len]
        action = find_next_action(snake_head, next_position)

        _, _, done, _ = env.step(action)
        num_step += 1
        env.render()
        time.sleep(FRAME_DELAY)

        if done:
            print(f"Game Finished: Score = {env.game.score}, Total steps = {num_step}")
            time.sleep(ROUND_DELAY)

if __name__ == "__main__":
    main()
