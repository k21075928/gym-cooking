import os
import pygame
import numpy as np
from utils.core import *
from misc.game.utils import *

graphics_dir = 'misc/game/graphics'
_image_library = {}
pygame.font.init() 
def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
        canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
        image = pygame.image.load(canonicalized_path)
        _image_library[path] = image
    return image


class Game:
    def __init__(self, world, sim_agents,rs1,rs2,arglist=None, play=False):
        self.mapGen = world
        self._running = True
        self.world = world
        self.sim_agents = sim_agents
        self.count_agent = self.sim_agents[0]
        self.play = play
        self.rs1=rs1
        self.rs2=rs2
        
        # Visual parameters
        self.scale = 80   # num pixels per tile
        self.holding_scale = 0.5
        self.container_scale = 0.7
        self.width = self.scale * self.world.width
        self.height = self.scale * self.world.height
        self.tile_size = (self.scale, self.scale)
        self.holding_size = tuple((self.holding_scale * np.asarray(self.tile_size)).astype(int))
        self.container_size = tuple((self.container_scale * np.asarray(self.tile_size)).astype(int))
        self.holding_container_size = tuple((self.container_scale * np.asarray(self.holding_size)).astype(int))
        self.UpdateDispenserCounter= 0
        self.item_delivery_timer = {}
        self.dqlmap=False
        if self.rs2 and arglist:
            if arglist.time:
                self.timer=arglist.time
            else:
                self.timer=60
            self.score=0
        if self.rs1 and arglist:
            if arglist.health:
                self.health=arglist.health
            else:
                self.health= 100


    def on_init(self):
        pygame.init()
        if self.play:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            # Create a hidden surface
            self.screen = pygame.Surface((self.width, self.height))
        self._running = True
        self.font = pygame.font.SysFont('arialttf.tff', 21)


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def decrease_health(self):
        if self.rs1:
            self.health = self.health - 1

    def increase_health(self,num):
        if self.rs1:
            self.health = self.health + num
    
    def decrease_time(self):
        if self.rs2:
            self.timer = self.timer - 1
    def increase_score(self,num):
        if self.rs2:
            self.score = self.score + num
    
    def get_health(self):
        if self.rs1:
            return self.health
        else:
            return 0
    
    def get_time(self):
        if self.rs2:
            return self.timer
        else:  
            return 0
    
    def get_score(self):
        if self.rs2:
            return self.score
        else:
            return 0

    def on_render(self):
        self.screen.fill(Color.FLOOR)
        objs = []

        
        
        
        # Draw gridsquares
        for o_list in self.world.objects.values():
            for o in o_list:
                if isinstance(o, GridSquare):
                    self.draw_gridsquare(o)
                elif o.is_held == False:
                    objs.append(o)
        
        # Draw objects not held by agents
        for o in objs:
            self.draw_object(o)

        # Draw agents and their holdings
        for agent in self.sim_agents:
            self.draw_agent(agent)

        if self.play:
            pygame.display.flip()
            pygame.display.update()

            if self.rs1 :
                health = self.font.render("health: "+str(self.health), True, (0, 0, 0))
                health_rect = health.get_rect(center=(self.width/2, self.height/2))

                self.screen.blit(health, health_rect)
            if self.rs2:
                timer = self.font.render("time: "+str(self.timer), True, (0, 0, 0))
                timer_rect = timer.get_rect(center=(self.width/2, self.height/2))

                self.screen.blit(timer, timer_rect)
                score = self.font.render("score: "+str(self.score), True, (0, 0, 0))
                score_rect = score.get_rect(center=(self.width/2, self.height/2+30))

                self.screen.blit(score, score_rect)
            
        if self.rs1:
            health = self.font.render("health: "+str(self.health), True, (0, 0, 0))
            health_rect = health.get_rect(center=(self.width/2, self.height/2))

            self.screen.blit(health, health_rect)
        if self.rs2:
            timer = self.font.render("time: "+str(self.timer), True, (0, 0, 0))
            timer_rect = timer.get_rect(center=(self.width/2, self.height/2))

            self.screen.blit(timer, timer_rect)
            score = self.font.render("score: "+str(self.score), True, (0, 0, 0))
            score_rect = score.get_rect(center=(self.width/2, self.height/2+30))

            self.screen.blit(score, score_rect)
            
        


    def draw_gridsquare(self, gs):
        if self.rs1:
            health = self.font.render("health: "+str(self.health), True, (0, 0, 0))
            self.screen.blit(health, (0,0))
            if self.item_delivery_timer and not self.dqlmap:
                y_offset = 15
                delivery_title = self.font.render("Delivery", True, (0, 0, 0))
                self.screen.blit(delivery_title, (0, y_offset))
                y_offset += 15
                for item, timer in self.item_delivery_timer.items():
                    item_delivery_info = self.font.render(f"{item}: {timer}", True, (0, 0, 0))
                    self.screen.blit(item_delivery_info, (0, y_offset))
                    y_offset += 15

        if self.rs2:
            timer = self.font.render("time: "+str(self.timer), True, (0, 0, 0))
            self.screen.blit(timer, (0,0))
            score = self.font.render("score: "+str(self.score), True, (0, 0, 0))
            self.screen.blit(score, (0,15))
            if self.item_delivery_timer  and not self.dqlmap:
                y_offset = 30
                delivery_title = self.font.render("Delivery", True, (0, 0, 0))
                self.screen.blit(delivery_title, (0, y_offset))
                y_offset += 15
                for item, timer in self.item_delivery_timer.items():
                    item_delivery_info = self.font.render(f"{item}: {timer}", True, (0, 0, 0))
                    self.screen.blit(item_delivery_info, (0, y_offset))
                    y_offset += 15


        sl = self.scaled_location(gs.location)
        fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)

        if isinstance(gs, Counter):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)

        elif isinstance(gs, Delivery):
            pygame.draw.rect(self.screen, Color.DELIVERY, fill)
            self.draw('delivery', self.tile_size, sl)

        elif isinstance(gs, Cutboard):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('cutboard', self.tile_size, sl)
        
        elif isinstance(gs, Stove):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw('stove', self.tile_size, sl)

        return

    def draw(self, path, size, location):
        image_path = '{}/{}.png'.format(graphics_dir, path)
        try:
            image = pygame.transform.scale(get_image(image_path), size)
            self.screen.blit(image, location)
        except pygame.error:
            return



    def draw_agent(self, agent):
        self.draw('agent-{}'.format(agent.color),
            self.tile_size, self.scaled_location(agent.location))
        self.draw_agent_object(agent.holding)

    def draw_agent_object(self, obj):
        # Holding shows up in bottom right corner.
        if obj is None: return
        if any([isinstance(c, Plate) for c in obj.contents]): 
            self.draw('Plate', self.holding_size, self.holding_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge('Plate')
                self.draw(obj.full_name, self.holding_container_size, self.holding_container_location(obj.location))
                obj.merge(plate)
        else:
            self.draw(obj.full_name, self.holding_size, self.holding_location(obj.location))

    def draw_object(self, obj):
        if obj is None: return
        if any([isinstance(c, Plate) for c in obj.contents]): 
            self.draw('Plate', self.tile_size, self.scaled_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge('Plate')
                self.draw(obj.full_name, self.container_size, self.container_location(obj.location))
                obj.merge(plate)
        else:
            self.draw(obj.full_name, self.tile_size, self.scaled_location(obj.location))

    def scaled_location(self, loc):
        """Return top-left corner of scaled location given coordinates loc, e.g. (3, 4)"""
        return tuple(self.scale * np.asarray(loc))

    def holding_location(self, loc):
        """Return top-left corner of location where agent holding will be drawn (bottom right corner) given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.holding_scale)).astype(int))

    def container_location(self, loc):
        """Return top-left corner of location where contained (i.e. plated) object will be drawn, given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.container_scale)/2).astype(int))

    def holding_container_location(self, loc):
        """Return top-left corner of location where contained, held object will be drawn given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        factor = (1-self.holding_scale) + (1-self.container_scale)/2*self.holding_scale
        return tuple((np.asarray(scaled_loc) + self.scale*factor).astype(int))


    def on_cleanup(self):
        # pygame.display.quit()
        pygame.quit()
