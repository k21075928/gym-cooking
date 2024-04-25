# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe
from recipe_planner.utils import *
from recipe_planner.recipe import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planning
import navigation_planner.utils as nav_utils

# Other core modules
from utils.interact import interact
from utils.world import World
from utils.core import *
from misc.game.gameimage import GameImage

import copy
import networkx as nx
import numpy as np
from itertools import combinations, permutations, product
from collections import namedtuple

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment

CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")


class DQLOvercookedEnvironment(OvercookedEnvironment):
    def __init__(self, arglist):
        super().__init__(arglist)
        self.x =0
        self.y =0
        self.predict = False
        self.item_refresh_rate = {
            "Plate": self.arglist.plate_refresh_time,
            "Lettuce": self.arglist.lettuce_refresh_time,
            "Tomato": self.arglist.tomato_refresh_time,
            "Onion": self.arglist.onion_refresh_time,
            "Chicken": self.arglist.chicken_refresh_time
            }
        self.reward =0

    def execute_navigation(self):
        for agent in self.sim_agents:
            objD =None
            objD=interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action
            if objD is not None:
                    self.isdelivered(objD)
                    self.reward +=50
                    self.isdone = True
                    objD = None

    def reset(self):
        self.world = DQLWorld(arglist=self.arglist)
        self.reward =0
        self.Initalworld = self.world
        self.recipes = []
        self.sim_agents = []
        self.agent_actions = {}
        self.t = 0

        # For visualizing episode.
        self.rep = []
        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        # Load world & distances.
        self.load_level(
                level=self.arglist.level,
                num_agents=self.arglist.num_agents)
        self.repOBS = np.zeros((self.x, self.y, 4))

        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        self.obs_tm1 = copy.copy(self)
        self.game = GameImage(
                    filename=self.filename,
                    world=self.world,
                    sim_agents=self.sim_agents,
                    record=False)
        if self.arglist.record or self.arglist.with_image_obs or self.predict:
            if self.predict:
                self.game = GameImage(
                    filename=self.filename,
                    world=self.world,
                    sim_agents=self.sim_agents,
                    record=True)
                self.game.on_init()
            else:
                self.game = GameImage(
                    filename=self.filename,
                    world=self.world,
                    sim_agents=self.sim_agents,
                    record=self.arglist.record)
                self.game.on_init()

            if self.arglist.record or self.predict:
                self.game.save_image_obs(self.t)

 

        return self.repOBS


    def refreshfromtimer(self):
        for item, rate in self.item_refresh_rate.items():
            refresh = self.t % rate
            if refresh == 0:
                self.refresh(item[0].lower())
                self.game.item_delivery_timer[item] = rate
            else:
                self.game.item_delivery_timer[item] = rate-refresh
    

    def step(self, action_dict):
        self.reward=0
        if 0 and self.t % 5 == 0:
            print("===============================")
            print("[Deep Q-Learning] @ STEP {}".format(self.t))
            print("===============================")
        if self.t==0:
            if self.arglist.rs1 or self.arglist.rs2:
                self.objInit()
                self.game.item_delivery_timer = {item: rate for item, rate in self.item_refresh_rate.items()}
                self.recipes = self.find_best_recipe()
        
        self.t += 1
       
        if 0 and (self.arglist.rs1 or self.arglist.rs2):
            self.refreshAll()
        # Choose actions, may not to fix sim_agents representation

        actions_available = [(0, 1), (0, -1), (-1, 0), (1, 0)]#
        for sim_agent in self.sim_agents:
            if sim_agent.name not in action_dict:
                print(f"No action provided for {sim_agent.name}, skipping this agent for this step.")
                sim_agent.action = (0, 0)
                continue
            if not isinstance(action_dict[sim_agent.name], tuple):
                sim_agent.action = actions_available[action_dict[sim_agent.name]]
            else:
                sim_agent.action = action_dict[sim_agent.name]
        
        # Execute.
        self.check_collisions()
        self.execute_navigation()
        if  (self.arglist.rs1 or self.arglist.rs2) and not self.arglist.level == "ResourceScarcityDQL":
            self.refreshfromtimer()

        # States, self.rewards, done
        self.recipes = self.find_best_recipe()
        self.all_subtasks = self.run_recipes()
        self.subtasks_left = self.all_subtasks
        done = self.done()
        

        self.reward += self._define_goal_state()

        self.update_display()

        # Record the image.
        if self.arglist.record or self.predict:
            self.game.save_image_obs(self.t)
        next_state = self.repOBS

        info = {
            "t": self.t,
            "done": done, "termination_info": self.termination_info
        }

        return next_state, self.reward, done, info
    def refreshAll(self):
        for item in self.item_refresh_rate.keys():
            self.refresh(item[0].lower())
            self.game.item_delivery_timer[item] = self.item_refresh_rate[item]

    def done(self):
        # Done if the episode maxes out
        if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                    self.arglist.max_num_timesteps)
            self.successful = False
            return True
        # assert any([isinstance(subtask, recipe.Deliver) for subtask in self.all_subtasks]), "no delivery subtask"

        # Done if subtask is completed.
        for subtask in self.all_subtasks:
            # Double check all goal_objs are at Delivery.
            if isinstance(subtask, recipe.Deliver):
                _, goal_obj = nav_utils.get_subtask_obj(subtask)

                delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
                goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                if not any([gol == delivery_loc for gol in goal_obj_locs]):
                    self.termination_info = ""
                    self.successful = False
                    return False
        if  (self.arglist.rs2) and self.arglist.level == "ResourceScarcityDQL":
            self.refreshAll()
        self.termination_info = "Terminating because all deliveries were completed"
        self.recipes = self.find_best_recipe()
        self.all_subtasks = self.run_recipes()
        self.subtasks_left = self.all_subtasks

        self.successful = True
        return True
    
    
    
    
    


    
    def find_best_recipe(self):
        object_counts = {
            "Plate": 0,
            "Tomato": 0,
            "Lettuce": 0,
            "Onion": 0,
            "Chicken": 0
        }
        recipes = []

        for obj in self.world.get_object_list():
            if isinstance(obj, Object):
                for object_type in object_counts.keys():
                    if obj.contains(object_type):
                        object_counts[object_type] += 1

        recipe_conditions = [
            (["Chicken", "Plate", "Tomato", "Lettuce"], ChickenSalad),
            (["Onion", "Plate", "Tomato", "Lettuce"], OnionSalad),
            (["Chicken", "Plate", "Tomato"], TomatoChicken),
            (["Chicken", "Plate", "Lettuce"], LettuceChicken),
            (["Onion", "Plate", "Tomato"], TomatoOnion),
            (["Onion", "Plate", "Lettuce"], OnionLettuce),
            (["Plate", "Tomato", "Lettuce"], Salad),
            (["Plate", "Chicken"], SimpleChicken),
            (["Plate", "Tomato"], SimpleTomato),
            (["Plate", "Lettuce"], SimpleLettuce),
            (["Plate", "Onion"], SimpleOnion)
        ]

        for condition, recipe in recipe_conditions:
            if all(object_counts[object_type] > 0 for object_type in condition):
                recipes.append(recipe())

        if not recipes:
            self.refreshRecipe = True

        return recipes
    
            
    
    def action_reduction(self, agent_name):
        actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        agent = next((a for a in self.sim_agents if a.name == agent_name), None)
        if not agent:
            return []

        if agent.holding:
            return self._actions_for_holding_agent(agent, actions)
        else:
            return self._actions_for_empty_agents(agent, actions)

    def _actions_for_holding_agent(self, agent, actions):
        actions_ = []
        for subtask in self.subtasks_left:
            if isinstance(subtask, recipe.Deliver):
                _, goal_obj = nav_utils.get_subtask_obj(subtask)
                if agent.holding == goal_obj:
                    # If the agent is holding the goal object, only keep actions that lead to a Delivery grid square
                    actions_ = [action for action in actions if isinstance(self.world.get_gridsquare_at(location=tuple(np.asarray(action) + np.asarray(agent.location))), Delivery)]
                else:
                    actions_.extend([action for action in actions if not isinstance(self.world.get_gridsquare_at(location=tuple(np.asarray(action) + np.asarray(agent.location))), Delivery)])
            else:
                actions_.extend(self._get_final_actions(agent, actions))
        return list(set(actions_))
    def _get_final_actions(self, agent, actions):
        final_actions = []
        for action in actions:
            next_location = tuple(np.asarray(agent.location) + np.asarray(action))
            all_objs = self.world.get_object_list()
            objs = list(filter(lambda o: o.location == next_location and (isinstance(o, Object) or isinstance(o, Food) or isinstance(o, Plate)), all_objs))
            if objs:
                all_objs = self.world.get_object_list()
                listUsed = list(filter(lambda o: o.location == next_location and (isinstance(o, Object) or isinstance(o, Food) or isinstance(o, Plate)), all_objs))
                if mergeable(agent.holding, listUsed[0]):
                    final_actions.append(action)
            elif not isinstance(self.world.get_gridsquare_at(location=tuple(np.asarray(action) + np.asarray(agent.location))), Delivery):
                final_actions.append(action)
        return final_actions

    def _actions_for_empty_agents(self, agent, actions):
        actions_ = []
        for action in actions:
            next_location = tuple(np.asarray(agent.location) + np.asarray(action))
            all_objs = self.world.get_object_list()
            objs = list(filter(lambda o: o.location == next_location and (isinstance(o, Object) or isinstance(o, Food) or isinstance(o, Plate)), all_objs))
            if objs:
                actions_.append(action)
            elif isinstance(self.world.get_gridsquare_at(location=next_location), Floor):
                actions_.append(action)
        if not actions_:
            actions_.append(random.choice(actions))
        return actions_












class DQLWorld(World):
    def __init__(self, arglist):
        super().__init__(arglist)
        self.repOBS = None 
