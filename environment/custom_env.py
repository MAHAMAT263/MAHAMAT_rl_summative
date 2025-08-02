from environment.rendering import render_grid
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class SeedDeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, use_fixed_map=True, render_mode=None):
        super(SeedDeliveryEnv, self).__init__()

        self.render_mode = render_mode  # ✅ Add this
        self.grid_size = 5
        self.max_steps = 60
        self.use_fixed_map = use_fixed_map

        # Now 7 channels: EMPTY, FARM, ROCK, DRYLAND, DEPOT, (reserved), AGENT
        self.observation_space = spaces.Box(low=0, high=1, shape=(8 * 5 * 5,), dtype=np.uint8)
        self.action_space = spaces.Discrete(5)

        self.agent_pos = None
        self.grid = None
        self.steps_taken = 0
        self.done = False
        self.has_seed = False

        # Entity codes
        self.EMPTY = 0
        self.FARM = 1
        self.ROCK = 2
        self.DRYLAND = 3
        self.DEPOT = 4
        self.AGENT = 6


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.steps_taken = 0
        self.done = False
        self.has_seed = False

        if self.use_fixed_map:
            # Deterministic setup for the environment
            self.grid[0, 0] = self.DEPOT
            self.depot_pos = (0, 0)
            self.grid[4, 4] = self.FARM
            self.farm_pos = (4, 4)
            self.grid[1, 1] = self.ROCK
            self.grid[3, 1] = self.DRYLAND

            # Randomize agent start position on any EMPTY cell
            while True:
                r, c = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                if self.grid[r, c] == self.EMPTY:
                    self.agent_pos = [r, c]
                    break

        else:
            # Randomized environment setup
            # Place Farm
            while True:
                farm_pos = (random.randint(0, 4), random.randint(0, 4))
                if self.grid[farm_pos] == self.EMPTY:
                    self.grid[farm_pos] = self.FARM
                    self.farm_pos = farm_pos
                    break

            # Place Depot
            while True:
                depot_pos = (random.randint(0, 4), random.randint(0, 4))
                if self.grid[depot_pos] == self.EMPTY:
                    self.grid[depot_pos] = self.DEPOT
                    self.depot_pos = depot_pos
                    break

            # Place Rocks
            for _ in range(3):
                while True:
                    r, c = random.randint(0, 4), random.randint(0, 4)
                    if self.grid[r, c] == self.EMPTY:
                        self.grid[r, c] = self.ROCK
                        break

            # Place Dry Land
            for _ in range(3):
                while True:
                    r, c = random.randint(0, 4), random.randint(0, 4)
                    if self.grid[r, c] == self.EMPTY:
                        self.grid[r, c] = self.DRYLAND
                        break

            # Place Agent
            while True:
                r, c = random.randint(0, 4), random.randint(0, 4)
                if self.grid[r, c] == self.EMPTY:
                    self.agent_pos = [r, c]
                    break

        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((8, self.grid_size, self.grid_size), dtype=np.uint8)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                tile = self.grid[i, j]
                obs[tile, i, j] = 1

        # Clear agent channel
        obs[self.AGENT] = 0
        obs[self.AGENT, self.agent_pos[0], self.agent_pos[1]] = 1

        if self.has_seed:
            obs[5, self.agent_pos[0], self.agent_pos[1]] = 1

        return obs.flatten()

    
    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        reward = -0.1  # Step penalty
        r, c = self.agent_pos

        # Move logic
        if action == 0 and r > 0 and self.grid[r - 1, c] != self.ROCK:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and r < self.grid_size - 1 and self.grid[r + 1, c] != self.ROCK:  # down
            self.agent_pos[0] += 1
        elif action == 2 and c > 0 and self.grid[r, c - 1] != self.ROCK:  # left
            self.agent_pos[1] -= 1
        elif action == 3 and c < self.grid_size - 1 and self.grid[r, c + 1] != self.ROCK:  # right
            self.agent_pos[1] += 1
        elif action == 4:  # deliver
            current_tile = self.grid[r, c]
            if current_tile == self.DEPOT and not self.has_seed:
                self.has_seed = True
                reward = +2  # Pick up
            elif current_tile == self.FARM:
                if self.has_seed:
                    reward = +10  # Delivery
                    self.has_seed = False
                    self.done = True  # ✅ End episode after delivery
                else:
                    reward = -1  # Tried delivering without seed
            else:
                reward = -0.5  # Invalid delivery

        # Penalty for stepping on dry land
        if self.grid[self.agent_pos[0], self.agent_pos[1]] == self.DRYLAND:
            reward -= 1

        # Reward shaping
        if self.has_seed:
            dist = abs(self.agent_pos[0] - self.farm_pos[0]) + abs(self.agent_pos[1] - self.farm_pos[1])
            reward += 1 / (dist + 1)
        else:
            dist = abs(self.agent_pos[0] - self.depot_pos[0]) + abs(self.agent_pos[1] - self.depot_pos[1])
            reward += 0.5 / (dist + 1)

        # Step counter
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}
    
    def render(self):
        if not hasattr(self, "surface"):
            pygame.init()
            self.surface = pygame.display.set_mode((500, 500))
            pygame.display.set_caption("Seed Delivery Environment")

        # Keep the Pygame window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()  # Calls env.close() to clean up

        obs = self._get_obs()
        obs_3d = obs.reshape(8, self.grid_size, self.grid_size)
        render_grid(obs_3d, self.surface)

        pygame.display.flip()


    def close(self):
        if hasattr(self, "surface"):
            pygame.quit()
            del self.surface
