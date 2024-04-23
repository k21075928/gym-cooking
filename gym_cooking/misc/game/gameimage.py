import pygame
import os
import numpy as np
from PIL import Image
from misc.game.game import Game
# from misc.game.utils import *


class GameImage(Game):
    def __init__(self, filename, world, sim_agents,arglist =None, rs1=False, rs2=False, record=False):
        Game.__init__(self, world, sim_agents, rs1, rs2, arglist=arglist)
        self.base_dir = 'misc/game/record/{}/Trial_'.format(filename)
        self.record = record
        self.game_record_dir = self.get_unique_dir()

    def get_unique_dir(self):
        i = 1
        while os.path.exists(self.base_dir + str(i)):
            i += 1
        return self.base_dir + str(i)

    def on_init(self):
        super().on_init()

        if self.record:
            # Make game_record folder if doesn't already exist
            if not os.path.exists(self.game_record_dir):
                os.makedirs(self.game_record_dir)

            # Clear game_record folder
            for f in os.listdir(self.game_record_dir):
                os.remove(os.path.join(self.game_record_dir, f))

    def get_image_obs(self):
        self.on_render()
        img_int = pygame.PixelArray(self.screen)
        img_rgb = np.zeros([img_int.shape[1], img_int.shape[0], 3], dtype=np.uint8)
        for i in range(img_int.shape[0]):
            for j in range(img_int.shape[1]):
                color = pygame.Color(img_int[i][j])
                img_rgb[j, i, 0] = color.g
                img_rgb[j, i, 1] = color.b
                img_rgb[j, i, 2] = color.r
        return img_rgb

    def save_image_obs(self, t):
        self.on_render()
        pygame.image.save(self.screen, '{}/t={:03d}.png'.format(self.game_record_dir, t))
