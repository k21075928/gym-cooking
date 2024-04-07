import gym
import numpy as np
from dqlagent import RealDQLAgent


class DQLMain:

    def __init__(self, env, arglist, dqlAgents):
        self.arglist = arglist
        self.env = env
        self.dqlAgents = dqlAgents

    def train(self):
        while self.env.alive():
            self.env.isdone=False
            while not self.env.isdone and self.env.alive():
                action_dict = {}

                for agent in self.dqlAgents:
                    action = agent.select_action(obs=self.env)
                    action_dict[agent.name] = action

                obs, reward, done, info, rsflag = self.env.step(action_dict=action_dict)

                for agent in self.dqlAgents:
                    agent.refresh_subtasks(world=self.env.world)
                    agent.all_done()
                    

                

    



