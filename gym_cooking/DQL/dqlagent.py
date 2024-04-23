# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe_utils
from recipe_planner.utils import *
from recipe_planner.recipe import *


# Navigation planner
from navigation_planner.planners.e2e_brtdp import E2E_BRTDP
import navigation_planner.utils as nav_utils
# Other core modules
from utils.core import Counter, Cutboard, Food, Plate, Object
from utils.utils import agent_settings

import numpy as np
import copy
from termcolor import colored as color
from collections import namedtuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ['blue', 'magenta', 'yellow', 'green']
import random
from collections import deque

class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen = 800000)
    def save(self, state):
        self.memory.append(state)
    def sample(self, size):
        temp= NotImplemented
        if size < len(self.memory):
            return random.sample(self.memory, size)
        return random.sample(self.memory, len(self.memory))

class NeuralNetworkModel(nn.Module):
    def __init__(self, state_sizes, name, arglist):
        super(NeuralNetworkModel, self).__init__()
        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        self.state_sizes = state_sizes
        self.model = self.torchmodel()
        self.targetModel = self.torchmodel()
        self.model_name = name
        self.arglist = arglist
        self.alpha = self.arglist.alphaDQL

        
    def torchmodel(self):
        model = nn.Sequential(
            nn.Linear(self.state_sizes, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )
        return model
    
    def save_model(self):
        directory = os.path.dirname(self.model_name)
        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), self.model_name)

    def load_weights_from_file(self):
        state_dict = torch.load(self.model_name)
        self.model.load_state_dict(state_dict)

    def train_model(self, input_data, target_data):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.alpha)
        input_data = torch.from_numpy(input_data).float()
        target_data = torch.from_numpy(target_data).float()
        for epoch in range(15):
            optimizer.zero_grad()
            outputs = self.model(input_data)
            loss = criterion(outputs, target_data)
            loss.backward()
            optimizer.step()
            
    def update_target(self):
        self.targetModel.load_state_dict(self.model.state_dict())
    
    def model_predict(self, state, target=False):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            if target:
                return self.targetModel(state)
            return self.model(state)
    def get_best_action(self, state, reduced_actions):
        """
        Get the action with the highest Q-value from the model's prediction.
        Args:
            state: The current state of the environment.
            reduced_actions: A list of actions that have been reduced that the agent should take.
            use_target_model: Whether to use the target model for predictions.
        Returns:
            The index of the action with the highest Q-value.
        """
        state = state.reshape(1, self.state_sizes)
        action_values = self.model_predict(state, target=False).flatten()
        legal_action_values = [(i, value) for i, value in enumerate(action_values) if i in reduced_actions]
        best_action = max(legal_action_values, key=lambda item: item[1])[0]

        return best_action
    
    
    

    # def max_Q_action(self, state, legalActions, target=False):
    #     actions = self.model_predict(state.reshape(1, self.state_sizes), target=target)

    #     finalList = actions.flatten()
    #     maxIndex = 1000
    #     maxValue = float("-inf")
    #     for i in range(len(finalList)):
    #         if i not in legalActions:
    #             continue
    #         else:
    #             if finalList[i] > maxValue:
    #                 maxIndex = i
    #     return maxIndex
    


class RealDQLAgent:
    def __init__(self, arglist, st_size, name, model_name):
        self.nnmodel = NeuralNetworkModel(st_size, model_name, arglist)
        self.memory = ReplayBuffer()
        self.arglist = arglist
        self.name = name
        self.st_size = st_size
        self.color = color
        self.actions_available = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        self.max_exploration_steps = self.arglist.max_exploration_steps
        self.max_exploration_rate = self.arglist.max_exploration_rate
        self.min_exploration_rate = self.arglist.min_exploration_rate
        self.discount_factor = self.arglist.discount_factor
        self.exploration_rate = self.arglist.exploration_rate
        self.count = 0
        self.lambda_ = self.arglist.lambda_
        self.alpha = self.arglist.alphaDQL
        self.batchSize = self.arglist.batchSize
        self.replay_memory=[]
        self.reduced_actions = []
        
    def load_model_trained(self):
        state_dict = torch.load(self.nnmodel.model_name)
        self.nnmodel.model.load_state_dict(state_dict)

    def predict(self, state):
        return self.nnmodel.get_best_action(state, self.reduced_actions)
    
    def save_transition(self, transition):
        self.replay_memory.append(transition)
        self.memory.save(transition)
        #Decaying the epsilon whenever an obersevation is made
        self.count += 1
        if self.count < self.max_exploration_steps:
            self.exploration_rate = self.min_exploration_rate + ((self.max_exploration_steps - self.count)/self.max_exploration_steps)*(self.max_exploration_rate - self.min_exploration_rate)  
    def agent_actions(self, legalActions):
        self.reduced_actions=[]
        for action in legalActions:
            self.reduced_actions.append(self.actions_available.index(action))

    def epsilon_greedy(self, state, legalActions=None):
        value = random.randint(0, 1)
        if self.reduced_actions == []:
            return False
        if value < self.exploration_rate:
            return random.choice(self.reduced_actions)
        else:
            return self.nnmodel.get_best_action(state, self.reduced_actions)
        
    def softmax_exploration(self, state, legalActions=None, temperature=1.0): #Possible implementation of this policy not completed
        if self.reduced_actions == []:
            return False
        q_values = [self.nnmodel.model_predict(state, action) for action in self.reduced_actions]
        exp_q_values = np.exp(np.array(q_values) / temperature)
        probabilities = exp_q_values / np.sum(exp_q_values)
        chosen_action = np.random.choice(self.reduced_actions, p=probabilities)

        return chosen_action

    def update_target(self):
        #Frequency of updates every time
        self.nnmodel.update_target()
    
    def update_q_values(self):
        sampled_batch = self.memory.sample(self.batchSize)
        states_list = []
        input_data = np.zeros((len(sampled_batch), self.st_size))
        target_data = np.zeros((len(sampled_batch), 4))
        prediction_errors = np.zeros(len(sampled_batch))
        for batch_item in sampled_batch:
            state = batch_item[0]
            states_list.append(state)
        states = np.array(states_list)
        predictions = self.nnmodel.model_predict(states)
        for i in range(len(sampled_batch)):
            state, selected_action, reward, done = sampled_batch[i][0], sampled_batch[i][1][self.name], sampled_batch[i][2], sampled_batch[i][4]
            reward = 0 if reward is None else reward
            prediction = predictions[i]
            old_value = prediction[selected_action]

            if done: 
                prediction[selected_action] = reward
            else: 
                td_target = reward
                eligibility_trace = 1
                for j in range(i+1, len(sampled_batch)):
                    td_target += self.discount_factor * eligibility_trace * sampled_batch[j][2]
                    eligibility_trace *= self.discount_factor * self.lambda_
                prediction[selected_action] = td_target
            
            input_data[i] = state
            target_data[i] = prediction
            prediction_errors[i] = np.abs(prediction[selected_action] - old_value)

        self.nnmodel.train_model(input_data, target_data)
    

    
    


            

    
