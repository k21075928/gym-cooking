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
    def __init__(self, filename, world, sim_agents,rs1,rs2, arglist=None):
        Game.__init__(self, world, sim_agents,rs1,rs2, play=True, arglist=arglist)
        self.filename = filename 
        self.steps = 0
        self.save_dir = 'misc/game/screenshots'
        self.item_refresh_rate = {
                    "Plate": arglist.plate_refresh_time,
                    "Lettuce": arglist.lettuce_refresh_time,
                    "Tomato": arglist.tomato_refresh_time,
                    "Onion": arglist.onion_refresh_time,
                    "Chicken": arglist.chicken_refresh_time
                    }
        self.item_delivery_timer = {item: rate for item, rate in self.item_refresh_rate.items()}
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.rs1 or self.rs2:
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

    def refresh(self, item):
        item_initial_locations = {
            "t": self.tomatoLocationInitial,
            "o": self.onionLocationInitial,
            "p": self.plateLocationInitial,
            "l": self.lettuceLocationInitial,
            "c": self.chickenLocationInitial
        }

        if item_initial_locations[item] is not None:
            for location in item_initial_locations[item]:
                if self.world.is_occupied(location):
                    return
                else:
                    self.world.remove(self.world.get_counter_at(location))
                    counter = Counter(location=location)
                    obj = Object(location, contents=RepToClass[item]())
                    counter.acquire(obj=obj)
                    self.world.insert(obj=counter)
                    self.world.insert(obj=obj)
                    self.has_state_changed_due_to_ingredient_respawn = True
        return
    def refreshAll(self):
        for item, rate in self.item_refresh_rate.items():
            refresh = self.steps % rate
            if refresh == 0:
                self.refresh(item[0].lower())
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if self.rs1 or self.rs2:
                self.steps+=1 
                self.decrease_health()
                self.refreshAll()
                for item in self.item_delivery_timer:
                    self.item_delivery_timer[item] -= 1
                    if self.item_delivery_timer[item] == 0:
                        self.item_delivery_timer[item] = self.item_refresh_rate[item]
            # Save current image
            if event.key == pygame.K_RETURN:
                image_name = '{}_{}.png'.format(self.filename, datetime.now().strftime('%m-%d-%y_%H-%M-%S'))
                pygame.image.save(self.screen, '{}/{}'.format(self.save_dir, image_name))
                print('just saved image {} to {}'.format(image_name, self.save_dir))
                return
            
            # Switch current agent
            if pygame.key.name(event.key) in "1234":
                try:
                    self.count_agent = self.sim_agents[int(pygame.key.name(event.key))-1]
                except:
                    pass
                return

            # Control current agent
            x, y = self.count_agent.location
            objD =None
            if event.key in KeyToTuple.keys():
                action = KeyToTuple[event.key]
                self.count_agent.action = action
                objD = interact(self.count_agent, self.world)
                if objD is not None and (self.rs1 or self.rs2):
                    self.isdelivered(objD)
                    objD = None
                if objD is not None and (self.rs1==False or self.rs2 ==False):
                    self.isdelivered(objD)
                    objD = None
                    
    def isdone(self):
        if self.rs2 and self.steps> 100:
            print("Terminating because timelimit is over "+ str(self.steps)+" steps")
            self._running = False
        if self.rs1:
            if self.health ==0 or self.health<0:
                print("Terminating because your guests starved to death at "+ str(self.steps)+" steps")
                self._running = False

    def isdelivered(self,obj):
        if (self.rs1==False and self.rs2 == False):
            self._running = False
        score = obj.full_name.count("-")
        meat = obj.full_name.count("Chicken")
        reward = 0
        if self.rs1 or self.rs2:
            if self.rs1:
                if meat>0:
                    self.increase_health(5*score+5)
                    reward = score +3
                else:
                    self.increase_health(5*score)
                    reward = score
            if self.rs2:
                """Make rewards exponential increase instead of Linear increase - To value higher scores for more complex recipes"""
                if meat>0:
                    self.increase_score(round(10 * (score ** 1.5) + 10))  
                    reward = score + 3
                else:
                    self.increase_score(round(10 * (score ** 1.5)))  
                    reward = score
            delivery = self.world.get_counter_at(obj.location)
            self.world.remove(obj)
            delivery.release()
        
             

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
                self.isdone()
            self.on_render()
        self.on_cleanup()


