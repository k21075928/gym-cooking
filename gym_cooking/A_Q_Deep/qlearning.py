import recipe_planner.utils as recipe
from delegation_planner.delegator import Delegator
from delegation_planner.utils import SubtaskAllocDistribution
from navigation_planner.utils import get_subtask_obj, get_subtask_action_obj, get_single_actions
from utils.interact import interact
from utils.utils import agent_settings

from collections import defaultdict, namedtuple
from itertools import permutations, product, combinations
import scipy as sp
import numpy as np
import copy

class Delegator:
    def __init__(self):
        pass
    def delegate(self, agents, tasks):
        raise NotImplementedError()
    def assign(self, agents, delegated_tasks):
        for i, a in enumerate(agents):
            a.tasks = delegated_tasks[i]

class Qlearning(Delegator):
    def __init__(self, agent_name, all_agent_names,
            model_type, planner, none_action_prob):
        self.name = 'Qlearning Delegator'
        self.agent_name = agent_name
        self.all_agent_names = all_agent_names
        self.probs = None
        self.model_type = model_type
        self.priors = 'uniform' if model_type == 'up' else 'spatial'
        self.planner = planner
        self.none_action_prob = none_action_prob



class DeepQlearning(Delegator):
    def __init__(self, agent_name, all_agent_names,
            model_type, planner, none_action_prob):
        self.name = 'DeepQlearning Delegator'
        self.agent_name = agent_name
        self.all_agent_names = all_agent_names
        self.probs = None
        self.model_type = model_type
        self.priors = 'uniform' if model_type == 'up' else 'spatial'
        self.planner = planner
        self.none_action_prob = none_action_prob