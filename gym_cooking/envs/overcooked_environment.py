# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe
from recipe_planner.recipe import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planning
import navigation_planner.utils as nav_utils

# Other core modules
from utils.interact import interact
from utils.world import World
from utils.core import *
from utils.agent import SimAgent
from misc.game.gameimage import GameImage
from utils.agent import COLORS

import copy
import networkx as nx
import numpy as np
from itertools import combinations, permutations, product
from collections import namedtuple

import gym
from gym import error, spaces, utils
from gym.utils import seeding


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")


class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist):
        self.predict=False
        self.Initalworld = None
        self.arglist = arglist
        self.t = 0
        self.set_filename()
        self.rs1=self.arglist.rs1
        self.rs2=self.arglist.rs2
        self.reward=0
        self.rsflag = False
        self.delivered=[]
        self.game = None
        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False
        self.x = 0
        self.y = 0

        self.plateLocationInitial= []
        self.tomatoLocationInitial= []
        self.lettuceLocationInitial= []
        self.onionLocationInitial= []
        self.chickenLocationInitial= []
        self.has_state_changed_due_to_ingredient_respawn = False
        self.item_refresh_rate = {
            "Plate": self.arglist.plate_refresh_time,
            "Lettuce": self.arglist.lettuce_refresh_time,
            "Tomato": self.arglist.tomato_refresh_time,
            "Onion": self.arglist.onion_refresh_time,
            "Chicken": self.arglist.chicken_refresh_time
            }
        self.item_delivery_timer = {item: rate for item, rate in self.item_refresh_rate.items()}

        

    def get_repr(self):
        return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances
        new_env.reward= self.reward
        new_env.plateLocationInitial = self.plateLocationInitial
        new_env.tomatoLocationInitial = self.tomatoLocationInitial
        new_env.lettuceLocationInitial = self.lettuceLocationInitial
        new_env.onionLocationInitial = self.onionLocationInitial
        new_env.chickenLocationInitial = self.chickenLocationInitial
        new_env.rsflag = self.rsflag 
        new_env.t = self.t
        new_env.delivered = self.delivered

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                        location=a.location,
                        desired_obj=None,
                        find_held_objects=True)
        return new_env

    def resetfornextround(self):
        self.recipes = []
        self.agent_actions = {}
        for obj in self.Initalworld.get_object_list():
            print(obj.name)
        # For visualizing episode.
        self.rep = []
        self.isdone = False

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False
        if self.arglist.dql:
            self.delivered =[]

        # Load world & distances.
        
        self.all_subtasks = self.run_recipes()
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        self.obs_tm1 = copy.copy(self)

        

        return copy.copy(self)
    
    def set_filename(self):
        self.filename = "{}_agents{}_seed{}".format(self.arglist.level,\
            self.arglist.num_agents, self.arglist.seed)
        model = ""
        if self.arglist.model1 is not None:
            model += "_model1-{}".format(self.arglist.model1)
        if self.arglist.model2 is not None:
            model += "_model2-{}".format(self.arglist.model2)
        if self.arglist.model3 is not None:
            model += "_model3-{}".format(self.arglist.model3)
        if self.arglist.model4 is not None:
            model += "_model4-{}".format(self.arglist.model4)
        self.filename += model
        if self.arglist.rs1:
            self.filename+="_resourceScarcityVersion1"
        if self.arglist.rs2:
            self.filename+="_resourceScarcityVersion2"
        if self.arglist.dql:
            self.filename+="_DQLVersion1"
            if self.arglist.unlimited:
                self.filename += "_Unlimited"
            else:
                self.filename += "_Training_{}".format(self.arglist.num_training)
        if self.predict:
            self.filename+="_PredictVersion"

    def load_level(self, level, num_agents):
        x = 0
        y = 0
        with open('utils/levels/{}.txt'.format(level), 'r') as file:
            # Mark the phases of reading.
            phase = 1
            for line in file:
                line = line.strip('\n')
                if line == '':
                    phase += 1

                # Phase 1: Read in kitchen map.
                elif phase == 1:
                    for x, rep in enumerate(line):
                        # Object, i.e. Tomato, Lettuce, Onion, or Plate, chicken.
                        if rep in 'tlopc':
                            counter = Counter(location=(x, y))
                            obj = Object(
                                    location=(x, y),
                                    contents=RepToClass[rep]())
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery, stove.
                        elif rep in RepToClass:
                            newobj = RepToClass[rep]((x, y))
                            self.world.objects.setdefault(newobj.name, []).append(newobj)
                        else:
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault('Floor', []).append(f)
                    y += 1
                # Phase 2: Read in recipe list.
                elif phase == 2:
                    if self.rs1 or self.rs2:
                        self.recipes=[]
                    else:
                        self.recipes.append(globals()[line]())
                # Phase 3: Read in agent locations (up to num_agents).
                elif phase == 3:
                    if len(self.sim_agents) < num_agents:
                        loc = line.split(' ')
                        sim_agent = SimAgent(
                                name='agent-'+str(len(self.sim_agents)+1),
                                id_color=COLORS[len(self.sim_agents)],
                                location=(int(loc[0]), int(loc[1])))
                        self.sim_agents.append(sim_agent)

        self.distances = {}
        self.world.width = x+1
        self.world.height = y
        self.world.perimeter = 2*(self.world.width + self.world.height)
        self.x = x+1
        self.y = y

    

    def reset(self):
        self.world = World(arglist=self.arglist)
        self.Initalworld = self.world
        self.recipes = []
        self.sim_agents = []
        self.agent_actions = {}
        self.t = 0
        self.plateLocationInitial= []
        self.tomatoLocationInitial= []
        self.lettuceLocationInitial= []
        self.onionLocationInitial= []
        self.chickenLocationInitial= []
        self.rsflag = False
        self.delivered =[]
        self.isdone = False

        for obj in self.Initalworld.get_object_list():
            print(obj.name)
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
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        self.obs_tm1 = copy.copy(self)

        if self.arglist.record or self.arglist.with_image_obs:
            self.game = GameImage(
                    filename=self.filename,
                    world=self.world,
                    sim_agents=self.sim_agents,
                    record=self.arglist.record,
                    rs1=self.arglist.rs1,rs2=self.arglist.rs2, arglist=self.arglist)
            self.game.on_init()
            if self.arglist.level == "ResourceScarcityDQL":
                self.game.dqlmap=True
            if self.arglist.record:
                self.game.save_image_obs(self.t)
        
        self.all_subtasks = self.run_recipes()
        return copy.copy(self)

    def close(self):
        return
    
    def objInit(self):
        for obj in self.Initalworld.get_object_list():
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


        
    def step(self, action_dict):
        if self.t==0:
            if self.arglist.rs1 or self.arglist.rs2:
                self.objInit()
                self.game.item_delivery_timer = {item: rate for item, rate in self.item_refresh_rate.items()}
                self.recipes = self.find_best_recipe()
                

        # Track internal environment info.
        self.t += 1
        print("===============================")
        print("[environment.step] @ TIMESTEP {}".format(self.t))
        print("===============================")
        if self.rs1:
            self.game.decrease_health()
        if self.rs2:
            self.game.decrease_time()
        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = action_dict[sim_agent.name]

        # Check collisions.
        self.check_collisions()
        self.obs_tm1 = copy.copy(self)

        if self.arglist.level == "ResourceScarcityDQL":
            self.done()
        if  (self.arglist.rs1 or self.arglist.rs2) and not self.arglist.level == "ResourceScarcityDQL":
            self.refreshAll()
        # Execute.
        self.execute_navigation()
        
        reward=self._define_goal_state()
        # Visualize.
        self.display()
        self.print_agents()
        if self.arglist.record:
            self.game.save_image_obs(self.t)

        # Get a plan-representation observation.
        new_obs = copy.copy(self)
        # Get an image observation
        image_obs = self.game.get_image_obs()
        
        #reward = self.reward()
        info = {"t": self.t, "obs": new_obs,
                "image_obs": image_obs,
                "done": self.isdone, "termination_info": self.termination_info}
        if self.rs1 or self.rs2:
            return new_obs, reward, self.isdone, info, self.rsflag
        else:
            return new_obs, reward, self.isdone, info, False
        
    def _define_goal_state(self):
        """Defining a goal state (termination condition on state) for subtask."""
        subtasks = self.run_recipes()
        reward = 0
        self.removed_object =[]
        for subtask in subtasks:
            for agent in self.sim_agents:
                if agent.holding is not None:
                    self.removed_object = agent.holding
            self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask)
            self.subtask_action_obj = nav_utils.get_subtask_action_obj(subtask)
            if subtask is None:
                self.is_goal_state = lambda h: True

            # Termination condition is when desired object is at a Deliver location.
            elif isinstance(subtask, recipe.Deliver):
                # Get current count of desired objects.
                self.cur_obj_count = len(list(filter(lambda o: o in set(self.world.get_all_object_locs(
                    self.subtask_action_obj)), self.world.get_object_locs(self.goal_obj, is_held=False))))
                self.has_more_obj = lambda x: int(x) > self.cur_obj_count
                self.is_goal_state = lambda h: self.has_more_obj(
                    len(list(filter(lambda o: o in set(self.world.get_all_object_locs(self.subtask_action_obj)),
                                    self.repr_to_env_dict[h].world.get_object_locs(self.goal_obj, is_held=False)))))

                if self.removed_object is not None and self.removed_object == self.goal_obj:
                    self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(list(filter(lambda o: o in set(self.world.get_all_object_locs(self.subtask_action_obj)),
                                        w.get_object_locs(self.goal_obj, is_held=False)))) + 1)
                else:
                    self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(list(filter(lambda o: o in set(self.world.get_all_object_locs(self.subtask_action_obj)),
                                        w.get_object_locs(obj=self.goal_obj, is_held=False)))))

                # Check if subtask is complete and add reward
                if self.is_subtask_complete(self.world):
                    reward += 1

            else:
                # Get current count of desired objects.
                self.cur_obj_count = len(self.world.get_all_object_locs(self.goal_obj))
                # Goal state is reached when the number of desired objects has increased.
                self.has_more_obj = lambda x: int(x) > self.cur_obj_count
                self.is_goal_state = lambda h: self.has_more_obj(
                    len(self.repr_to_env_dict[h].world.get_all_object_locs(self.goal_obj)))

                if self.removed_object is not None and self.removed_object == self.goal_obj:
                    self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(w.get_all_object_locs(self.goal_obj)) + 1)
                else:
                    self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(w.get_all_object_locs(self.goal_obj)))

                # Check if subtask is complete and add reward
                if self.is_subtask_complete(self.world):
                    reward += 1

        return reward
    def alive(self):
        if self.rs1:
            self.termination_info = "Terminating because you guest starved to death at {}".format(self.t)
            return self.game.health>0 
        if self.rs2:
            self.termination_info = "Terminating because you ran out of time {}".format(self.t)
            if self.game is not None:
                return self.game.timer>0
            else:
                return True
    def refreshAll(self):
        for item, rate in self.item_refresh_rate.items():
            refresh = self.t % rate
            if refresh == 0:
                self.refresh(item[0].lower())
                self.game.item_delivery_timer[item] = rate
            else:
                self.game.item_delivery_timer[item] = rate-refresh

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
    def done(self):
        # if self.rs1==False and self.rs2 == False:
             # Done if the episode maxes out
        # if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
        #     self.termination_info = "Terminating because passed {} timesteps".format(
        #             self.arglist.max_num_timesteps)
        #     self.successful = False
        #     return True

        if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                    self.arglist.max_num_timesteps)
            self.successful = False
            return True

        assert any([isinstance(subtask, recipe.Deliver) for subtask in self.all_subtasks]), "no delivery subtask"

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
        
        self.termination_info = "Terminating because all deliveries were completed"
        self.successful = True
        return True
    
    
        # else:
        #     print("RSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
        #     if self.game.health ==0 or self.game.health<0:
        #         self.termination_info = "Terminating because you guest starved to death at {}".format(
        #                 self.t)
        #         if self.t<100 or self.reward==0:
        #             self.successful = False
        #             return False
        #         self.successful = True
        #         return True

        #     # Done if subtask is completed.
        #     for subtask in self.all_subtasks:
        #         # Double check all goal_objs are at Delivery.
        #         if isinstance(subtask, recipe.Deliver):
        #             _, goal_obj = nav_utils.get_subtask_obj(subtask)
        #             delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
        #             goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
        #             if not any([gol == delivery_loc for gol in goal_obj_locs]):
        #                 print("Completed Recipe onto the next :)")
        #                 self.rsflag= True
        #                 self.termination_info = "1 recipe completed"
        #                 self.successful = True
        #     if(self.all_subtasks is None):
        #         self.termination_info = "No Recipes can be completed"
        #         self.successful = True
        #         return True
        #     if self.rs2 and self.t>99:
        #         self.termination_info = "Terminating because you ran out of time {}".format(
        #                 self.t)
        #         self.successful = False
        #         return True

    def refreshInstant(self):
        for item in self.item_refresh_rate.keys():
            self.refresh(item[0].lower())


    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)


    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]

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
            (["Chicken", "Plate", "Tomato", "Lettuce"], ChickenSalad, 30),
            (["Onion", "Plate", "Tomato", "Lettuce"], OnionSalad, 30),
            (["Chicken", "Plate", "Tomato"], TomatoChicken, 20),
            (["Chicken", "Plate", "Lettuce"], LettuceChicken, 20),
            (["Onion", "Plate", "Tomato"], TomatoOnion, 20),
            (["Onion", "Plate", "Lettuce"], OnionLettuce, 20),
            (["Plate", "Tomato", "Lettuce"], Salad, 20),
            (["Plate", "Chicken"], SimpleChicken, 0),
            (["Plate", "Tomato"], SimpleTomato, 0),
            (["Plate", "Lettuce"], SimpleLettuce, 0),
            (["Plate", "Onion"], SimpleOnion, 0)
        ]

        for condition, recipe, min_health_time in recipe_conditions:
            if all(object_counts[object_type] > 0 for object_type in condition) and (self.game.get_health() > min_health_time or self.game.get_time() > min_health_time):
                recipes.append(recipe())

        if not recipes:
            self.refreshRecipe = True

        return recipes
    
    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        if self.rs1 or self.rs2:
            self.recipes = self.find_best_recipe()
        self.sw = STRIPSWorld(world=self.world, recipes=self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]
        # print('Subtasks:', all_subtasks, '\n')
        return all_subtasks

    def get_AB_locs_given_objs(self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Returns list of locations relevant for subtask's Merge operator.

        See Merge operator formalism in our paper, under Fig. 11:
        https://arxiv.org/pdf/2003.11778.pdf"""

        # For Merge operator on Chop subtasks, we look at objects that can be
        # chopped and the cutting board objects.
        if isinstance(subtask, recipe.Chop):
            # A: Object that can be chopped.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(map(lambda a: a.location,\
                list(filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))

            # B: Cutboard objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)
        
        elif isinstance(subtask, recipe.Cook):
            # A: Object that can be cooked.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(map(lambda a: a.location,\
                list(filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))

            # B: stove objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

        # For Merge operator on Deliver subtasks, we look at objects that can be
        # delivered and the Delivery object.
        elif isinstance(subtask, recipe.Deliver):
            # B: Delivery objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

            # A: Object that can be delivered.
            A_locs = self.world.get_object_locs(
                    obj=start_obj, is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))
            A_locs = list(filter(lambda a: a not in B_locs, A_locs))

        # For Merge operator on Merge subtasks, we look at objects that can be
        # combined together. These objects are all ingredient objects (e.g. Tomato, Lettuce).
        elif isinstance(subtask, recipe.Merge):
            A_locs = self.world.get_object_locs(
                    obj=start_obj[0], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[0], self.sim_agents))))
            B_locs = self.world.get_object_locs(
                    obj=start_obj[1], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[1], self.sim_agents))))

        else:
            return [], []

        return A_locs, B_locs

    def get_lower_bound_for_subtask_given_objs(
            self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Return the lower bound distance (shortest path) under this subtask between objects."""
        assert len(subtask_agent_names) <= 2, 'passed in {} agents but can only do 1 or 2'.format(len(agents))

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                # Check for whether the agent is holding something.
                if agent.holding is not None:
                    if isinstance(subtask, recipe.Merge):
                        continue
                    else:
                        if agent.holding != start_obj and agent.holding != goal_obj:
                            # Add one "distance"-unit cost
                            holding_penalty += 1.0
        # Account for two-agents where we DON'T want to overpenalize.
        holding_penalty = min(holding_penalty, 1)

        # Get current agent locations.
        agent_locs = [agent.location for agent in list(filter(lambda a: a.name in subtask_agent_names, self.sim_agents))]
        A_locs, B_locs = self.get_AB_locs_given_objs(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                start_obj=start_obj,
                goal_obj=goal_obj,
                subtask_action_obj=subtask_action_obj)

        # Add together distance and holding_penalty.
        return self.world.get_lower_bound_between(
                subtask=subtask,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs)) + holding_penalty

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places.
        elif ((agent1_loc == agent2_next_loc) and
                (agent2_loc == agent1_next_loc)):
            execute[0] = False
            execute[1] = False
        return execute

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                    agent1_loc=agent_i.location,
                    agent2_loc=agent_j.location,
                    agent1_action=agent_i.action,
                    agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                        time=self.t,
                        agent_names=[agent_i.name, agent_j.name],
                        agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)

        # print('\nexecute array is:', execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)
            # print("{} has action {}".format(color(agent.name, agent.color), agent.action))

    def execute_navigation(self):
        for agent in self.sim_agents:
            objD =None
            objD=interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action
            if objD is not None:
                    self.isdelivered(objD)
                    self.isdone = True
                    objD = None

    def isdelivered(self,obj):
        score = obj.full_name.count("-")
        meat = obj.full_name.count("Chicken")
        reward = 0
        if self.rs1 or self.rs2:
            if self.rs1:
                if meat>0:
                    self.game.increase_health(5*score+5)
                    reward = score +3
                else:
                    self.game.increase_health(5*score)
                    reward = score
            if self.rs2:
                """Make rewards exponential increase instead of Linear increase - To value higher scores for more complex recipes"""
                if meat>0:
                    self.game.increase_score(round(10 * (score ** 1.5) + 10))  
                    reward = score + 3
                else:
                    self.game.increase_score(round(10 * (score ** 1.5)))  
                    reward = score
            self.delivered.append(obj.name)
            delivery = self.world.get_counter_at(obj.location)
            self.world.remove(obj)
            delivery.release()
            self.reward += reward
            print(self.arglist.level )
            if self.arglist.level == "ResourceScarcityDQL":
                self.refreshInstant()


    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [name for name in self.world.objects if "Supply" in name or "Counter" in name or "Delivery" in name or "Cut" in name or "Stove" in name]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                # Possible edges to approach source and destination.
                source_edges = [(0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(self.world.reachability_graph, (source.location,source_edge), (destination.location, dest_edge))
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances
