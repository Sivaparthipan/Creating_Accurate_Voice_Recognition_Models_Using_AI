import random
import gymnasium
import numpy as np
import pygame
from pygame.locals import *
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

# Snake Game Environment
class SnakeGameEnv(Env):
    def __init__(self):
        super(SnakeGameEnv, self).__init__()
        self.display_width = 800
        self.display_height = 600
        self.snake_block = 10
        self.snake_speed = 15
        
        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.array([0, 0, 0, 0]), high=np.array([self.display_width, self.display_height, self.display_width, self.display_height]), dtype=np.float32)
        
        self.seed()
        self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.x1 = self.display_width / 2
        self.y1 = self.display_height / 2
        self.x1_change = 0
        self.y1_change = 0
        self.snake_List = []
        self.Length_of_snake = 1
        self.foodx = round(random.randrange(0, self.display_width - self.snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, self.display_height - self.snake_block) / 10.0) * 10.0
        self.game_over = False
        return self._get_state(), {}

    def _get_state(self):
        state = np.array([self.x1, self.y1, self.foodx, self.foody], dtype=np.float32)
        return state

    def step(self, action):
        reward = 0
        if action == 0:
            self.x1_change = -self.snake_block
            self.y1_change = 0
        elif action == 1:
            self.x1_change = self.snake_block
            self.y1_change = 0
        elif action == 2:
            self.y1_change = -self.snake_block
            self.x1_change = 0
        elif action == 3:
            self.y1_change = self.snake_block
            self.x1_change = 0

        self.x1 += self.x1_change
        self.y1 += self.y1_change

        if self.x1 >= self.display_width or self.x1 < 0 or self.y1 >= self.display_height or self.y1 < 0:
            self.game_over = True
            reward = -10
        else:
            reward = -0.1
        
        snake_Head = [self.x1, self.y1]
        self.snake_List.append(snake_Head)
        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]
        
        for segment in self.snake_List[:-1]:
            if segment == snake_Head:
                self.game_over = True
                reward = -10

        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, self.display_width - self.snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, self.display_height - self.snake_block) / 10.0) * 10.0
            self.Length_of_snake += 1
            reward = 10

        state = self._get_state()
        return state, reward, self.game_over, {}

    def render(self):
        pygame.init()
        dis = pygame.display.set_mode((self.display_width, self.display_height))
        clock = pygame.time.Clock()
        while not self.game_over:
            dis.fill((255, 255, 255))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True

            pygame.draw.rect(dis, (0, 0, 255), [self.x1, self.y1, self.snake_block, self.snake_block])
            pygame.draw.rect(dis, (255, 0, 0), [self.foodx, self.foody, self.snake_block, self.snake_block])
            for segment in self.snake_List:
                pygame.draw.rect(dis, (0, 0, 0), [segment[0], segment[1], self.snake_block, self.snake_block])

            pygame.display.update()
            clock.tick(self.snake_speed)

        pygame.quit()

env = DummyVecEnv([lambda: SnakeGameEnv()])

# Train the model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# Save the model
model.save("../models/dqn_snake_model")

# Load the model and test
model = DQN.load("../models/dqn_snake_model")

# Test the trained model
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break
