import time
import random
import pygame
import numpy as np
import sys

class SnakeGame:
    def __init__(self):
        self.board_size = 11
        self.cell_size = 40
        self.width = self.board_size * self.cell_size
        self.height = self.board_size * self.cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")

        self.reset()
        self.clock = pygame.time.Clock()

    def reset(self):
        self.snake = [(self.board_size // 2, self.board_size // 2)]
        self.direction = "DOWN"
        self.food = self._generate_food()
        return self._get_state()

    def step(self, action):
        # Update direction
        self.update_direction(action)

        # Move snake
        row, col = self.snake[0]
        if self.direction == "UP":
            row -= 1
        elif self.direction == "DOWN":
            row += 1
        elif self.direction == "LEFT":
            col -= 1
        elif self.direction == "RIGHT":
            col += 1

        # Check if snake ate food
        if (row, col) == self.food:
            self.food = self._generate_food()
        else:
            self.snake.pop()
        
        self.snake.insert(0, (row, col))

        # Check if snake collided with itself or the wall
        done = (
            self.snake[0] in self.snake[1:]
            or row < 0
            or row >= self.board_size
            or col < 0
            or col >= self.board_size
        )

        # Check if snake collided with itself or the wall
        done = (self.snake[0] in self.snake[1:] or row < 0 or row >= self.board_size or col < 0 or col >= self.board_size)

        # Calculate reward
        if done:
            reward = -1
        elif self.snake[0] == self.food:
            reward = 1
        else:
            reward = 0

        return self._get_state(), reward, done

    def update_direction(self, action):
        if action == 0:  # Move up
            if self.direction != "DOWN":
                self.direction = "UP"
        elif action == 1:  # Move left
            if self.direction != "RIGHT":
                self.direction = "LEFT"
        elif action == 2:  # Move right
            if self.direction != "LEFT":
                self.direction = "RIGHT"
        elif action == 3:  # Move down
            if self.direction != "UP":
                self.direction = "DOWN"
        # If action is anything else, do nothing.

    def _generate_food(self):
        food = (random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1))
        while food in self.snake:
            food = (random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1))
        return food

    def _get_state(self):
        state = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        for r, c in self.snake:
            state[r, c] = 1
        state[self.food] = 2
        return state.flatten()

    def render(self):
        self.screen.fill((0, 0, 0))

        # Draw snake
        for r, c in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

        # Draw food
        r, c = self.food
        pygame.draw.rect(self.screen, (255, 0, 0), (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    game = SnakeGame()
    game.reset()
    game.render()
    done = False
    action = -1

    update_interval = 0.2
    start_time = time.time()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 3
                elif event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 2
        
        if time.time() - start_time >= update_interval:
            _, _, done = game.step(action)
            game.render()
            start_time = time.time()

        if done:
            game.reset()
            game.render()
            action = -1
        
        pygame.time.wait(10)
