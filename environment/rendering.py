import pygame
import numpy as np

TILE_SIZE = 100
MARGIN = 6

ENTITY_COLORS = {
    0: (255, 255, 255),  # EMPTY - white
    1: (0, 255, 0),      # FARM - bright green
    2: (100, 100, 100),  # ROCK - grey
    3: (153, 101, 21),   # DRYLAND - brown
    4: (255, 255, 102),  # DEPOT - light yellow
    6: (0, 0, 255),      # AGENT - blue
}

def render_grid(obs, surface):
    surface.fill((0, 0, 0))  # background
    for r in range(obs.shape[1]):
        for c in range(obs.shape[2]):
            cell = obs[:, r, c]

            # ✅ If agent is present, render it regardless of other entities
            if cell[6] == 1:
                color = ENTITY_COLORS[6]
            else:
                entity = np.argmax(cell)
                color = ENTITY_COLORS.get(entity, (255, 255, 255))

            rect = pygame.Rect(
                c * TILE_SIZE + MARGIN,
                r * TILE_SIZE + MARGIN,
                TILE_SIZE - 2 * MARGIN,
                TILE_SIZE - 2 * MARGIN,
            )
            pygame.draw.rect(surface, color, rect)