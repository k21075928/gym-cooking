# modules for game
from misc.game.game import Game
from misc.game.utils import *
from utils.core import *
from utils.interact import interact

# helpers
import pygame
import numpy as np
import argparse
from collections import defaultdict
from random import randrange
import os
from datetime import datetime


class GamePlay(Game):
    def __init__(self, filename, world, sim_agents):
        Game.__init__(self, world, sim_agents, play=True)
        self.filename = filename 
        self.steps = 0
        self.save_dir = 'misc/game/screenshots'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.plateLocationInitial= []
        self.tomatoLocationInitial= []
        self.chickenLocationInitial= []
        self.lettuceLocationInitial= []
        self.onionLocationInitial= []
        self.deliveryLocation= []
        self.counter=0
        self.Initalworld = world
        for obj in self.Initalworld.get_object_list():
            if isinstance(obj, Delivery):
                self.deliveryLocation.append(obj.location)
            if isinstance(obj, Object):
                if obj.contains("Plate"):
                    self.plateLocationInitial.append(obj.location)
                if obj.contains("Tomato"):
                    self.tomatoLocationInitial.append(obj.location)
                if obj.contains("Lettuce"):
                    self.lettuceLocationInitial.append(obj.location)
                if obj.contains("Onion"):
                    self.onionLocationInitial.append(obj.location)
                if obj.contains("Chicken"):
                    self.chickenLocationInitial.append(obj.location)

        # tally up all gridsquare types
        self.gridsquares = []
        self.gridsquare_types = defaultdict(set) # {type: set of coordinates of that type}
        for name, gridsquares in self.world.objects.items():
            for gridsquare in gridsquares:
                self.gridsquares.append(gridsquare)
                self.gridsquare_types[name].add(gridsquare.location)

    def refresh(self,item):             
        if item =="t" and self.tomatoLocationInitial is not None:
            for location in self.tomatoLocationInitial:
                if self.world.is_occupied(location):
                    return
                else:
                    obj = Object(location,contents=RepToClass["t"]())
                    self.world.insert(obj=obj)
        if item =="o" and  self.onionLocationInitial is not None:
            for location in self.onionLocationInitial:
                if self.world.is_occupied(location):
                    return
                else:
                    obj = Object(location,contents=RepToClass["o"]())
                    self.world.insert(obj=obj)
        if item =="p" and  self.plateLocationInitial is not None:
            for location in self.plateLocationInitial:
                if self.world.is_occupied(location):
                    return
                else:
                    obj = Object(location,contents=RepToClass["p"]())
                    self.world.insert(obj=obj)
        if item =="l" and  self.lettuceLocationInitial is not None:
            for location in self.lettuceLocationInitial:
                if self.world.is_occupied(location):
                    return
                else:
                    obj = Object(location,contents=RepToClass["l"]())
                    self.world.insert(obj=obj)
        if item =="c" and  self.chickenLocationInitial is not None:
            for location in self.chickenLocationInitial:
                if self.world.is_occupied(location):
                    return
                else:
                    obj = Object(location,contents=RepToClass["c"]())
                    self.world.insert(obj=obj)
        return

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            self.steps+=1 
            self.decrease_health()
            if (self.steps % 30==0):
                self.refresh("t")
            if (self.steps % 30==0):
                self.refresh("p")
            if (self.steps % 30==0):
                self.refresh("l")
            if (self.steps % 30==0):
                self.refresh("o")
            # Save current image
            if event.key == pygame.K_RETURN:
                image_name = '{}_{}.png'.format(self.filename, datetime.now().strftime('%m-%d-%y_%H-%M-%S'))
                pygame.image.save(self.screen, '{}/{}'.format(self.save_dir, image_name))
                print('just saved image {} to {}'.format(image_name, self.save_dir))
                return
            
            # Switch current agent
            if pygame.key.name(event.key) in "1234":
                try:
                    self.current_agent = self.sim_agents[int(pygame.key.name(event.key))-1]
                except:
                    pass
                return

            # Control current agent
            x, y = self.current_agent.location
            objD =None
            if event.key in KeyToTuple.keys():
                action = KeyToTuple[event.key]
                self.current_agent.action = action
                objD = interact(self.current_agent, self.world)
                if objD is not None:
                    self.isdelivered(objD)
                    objD = None

    def isdone(self):
        if self.health ==0 or self.health<0:
            print("Terminating because your guests starved to death at "+ str(self.steps)+" steps")
            self._running = False
    def isdelivered(self,obj):
        score = obj.full_name.count("-")
        self.increase_health(20*score)
        self.world.remove(obj)
             

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
                self.isdone()
            self.on_render()
        self.on_cleanup()


