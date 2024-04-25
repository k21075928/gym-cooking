# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

import numpy as np
import random
import argparse
from collections import namedtuple
from DQL.dqlagent import RealDQLAgent
import gym
import csv
import os

def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=60, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=80, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")

    # Resource Scarcity Versions
    parser.add_argument("--rs1", action="store_true", default=False, help="Resource Scarcity Version 1 ")
    parser.add_argument("--health", type=int, default=45, help="Inital health of Environment")
    parser.add_argument("--rs2", action="store_true", default=False, help="Resource Scarcity Version 2 ")
    parser.add_argument("--time", type=int, default=60, help="Time Limit of Environment")
    parser.add_argument("--onion_refresh_time", type=int, default=25, help="Refresh time for Onion")
    parser.add_argument("--plate_refresh_time", type=int, default=10, help="Refresh time for Plate")
    parser.add_argument("--lettuce_refresh_time", type=int, default=15, help="Refresh time for Lettuce")
    parser.add_argument("--tomato_refresh_time", type=int, default=10, help="Refresh time for Tomato")
    parser.add_argument("--chicken_refresh_time", type=int, default=30, help="Refresh time for Chicken")

    #Deep Q Learning
    parser.add_argument("--dql", action="store_true", default=False, help="Deep Q Learning") 
    parser.add_argument("--max_exploration_steps", type=int, default=100000, help="Max_exploration_steps")
    parser.add_argument("--max_exploration_rate", type=float, default=0.95, help="Max_exploration_rate")
    parser.add_argument("--min_exploration_rate", type=float, default=0.02, help="Min_exploration_rate")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount_factor")
    parser.add_argument("--exploration_rate", type=float, default=0.8, help="Exploration_rate")
    parser.add_argument("--lambda_", type=float, default=1, help="Lambda")
    parser.add_argument("--alphaDQL", type=float, default=0.0005, help="Alpha")
    parser.add_argument("--batchSize", type=int, default=128, help="BatchSize")
    parser.add_argument("--num_training", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--max_timestep", type=int, default=60, help="Maximum number of timesteps per episode")
    parser.add_argument("--unlimited", action="store_true", default=False, help="unbounded training episodes") 
    parser.add_argument("--record_data", action="store_true", default=False, help="Record Data") 
    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
#change here

def initialize_agents(arglist, state_size=0, nnmodel=None):
    real_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        index = 0#
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())
                
            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(real_agents) < arglist.num_agents:
                    if arglist.dql:
                        loc = line.split(' ')
                        real_agent = RealDQLAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            model_name=nnmodel[index],
                            st_size=state_size)
                        real_agents.append(real_agent)
                        index += 1
                    else:
                        loc = line.split(' ')
                        real_agent = RealAgent(
                                arglist=arglist,
                                name='agent-'+str(len(real_agents)+1),
                                id_color=COLORS[len(real_agents)],
                                recipes=recipes)
                        real_agents.append(real_agent)

    return real_agents
def main_loop(arglist):
    """The main loop for running experiments."""
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()
    csv_filename = create_csv_filename(arglist)
    accumulative_reward = 0
    if arglist.rs1 or arglist.rs2:
        bag = Bag(arglist=arglist, filename=env.filename)
        bag.set_recipe(recipe_subtasks=env.all_subtasks)
        while env.alive():  # Keep running until the environment is done
            real_agents = initialize_agents(arglist=arglist)
            env.isdone=False
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                while not env.isdone and env.alive():
                    action_dict = {}
                    for agent in real_agents:
                        action = agent.select_action(obs=obs)
                        action_dict[agent.name] = action
                    obs, reward, done, info, rsflag = env.step(action_dict=action_dict)
                    
                    if arglist.record_data:
                        accumulative_reward += reward
                        writer.writerow([env.t, accumulative_reward])
                    
                    # Agents
                    for agent in real_agents:
                        agent.refresh_subtasks(world=env.world)
                        agent.all_done()
                    bag.add_status(cur_time=info['t'], real_agents=real_agents)
                    

                # Saving info
                    
            bag.get_delivered(env.delivered)
            if arglist.rs2:
                bag.get_score(env.game.score)

            # Saving final information before saving pkl file
            bag.set_collisions(collisions=env.collisions)
            bag.set_termination(termination_info=env.termination_info,
                    successful=env.successful)
    else:

        # game = GameVisualize(env)
        real_agents = initialize_agents(arglist=arglist)
        # Info bag for saving pkl files
        bag = Bag(arglist=arglist, filename=env.filename)
        bag.set_recipe(recipe_subtasks=env.all_subtasks)
        delivered=[]
        with open(csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            while not env.done() or not env.isdone:
                action_dict = {}

                for agent in real_agents:
                    action = agent.select_action(obs=obs)
                    action_dict[agent.name] = action

                obs, reward, done, info, rsflag = env.step(action_dict=action_dict)
                print("info", info)
                #env.get_reward(real_agents=real_agents)

                # Agents
                for agent in real_agents:
                    agent.refresh_subtasks(world=env.world)
                    agent.all_done()

                # Saving info
                bag.add_status(cur_time=info['t'], real_agents=real_agents)


                if arglist.record_data:
                    accumulative_reward += reward
                    writer.writerow([env.t, accumulative_reward])
            bag.set_collisions(collisions=env.collisions)
            bag.set_termination(termination_info=env.termination_info,
            successful=env.successful)
            print("Delivered:", env.delivered)


def create_csv_filename(arglist):
    csv_filename = f"rewards_level{arglist.level}_model1{arglist.model1}"
    csv_filename = "{}_agents{}_seed{}".format(arglist.level,
            arglist.num_agents, arglist.seed)
    csv_filename = "Results/{}".format(csv_filename)
    directory = os.path.dirname(csv_filename)
    os.makedirs(directory, exist_ok=True)
    model = ""
    if arglist.model1 is not None:
        model += "_model1-{}".format(arglist.model1)
    if arglist.model2 is not None:
        model += "_model2-{}".format(arglist.model2)
    if arglist.model3 is not None:
        model += "_model3-{}".format(arglist.model3)
    if arglist.model4 is not None:
        model += "_model4-{}".format(arglist.model4)
    csv_filename += model
    if arglist.rs1:
        csv_filename+="_resourceScarcityVersion1"
    if arglist.rs2:
        csv_filename+="_resourceScarcityVersion2"
    if arglist.dql:
        csv_filename+="_DQLVersion1"
        csv_filename += "_Training_{}".format(arglist.num_training)
    csv_filename += ".csv"
    if arglist.record_data:
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", model])
    return csv_filename


def dqlMainLoop(arglist):
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:DQLovercookedEnv-v0", arglist=arglist)
    env.reset()
    csv_filename = create_csv_filename(arglist)
    if not arglist.rs2:
        filename1 = "agent_1-dql-level_{}-time{}_num_episodes_{}.h5".format(
            arglist.level,
            arglist.max_timestep, arglist.num_training
        )
        filename2 = "agent_2-dql-level_{}-time{}_num_episodes_{}.h5".format(
            arglist.level,
            arglist.max_timestep, arglist.num_training
        )
    else:
        filename1 = "agent_1-dql-level_{}-time{}_num_episodes_{}_ResourceScarcityVersion2.h5".format(
            arglist.level,
            arglist.max_timestep, arglist.num_training
        )
        filename2 = "agent_2-dql-level_{}-time{}_num_episodes_{}_ResourceScarcityVersion2.h5".format(
            arglist.level,
            arglist.max_timestep, arglist.num_training
        )
    model_file=['./DQL/DQLAgentTraining/{}'.format(filename1), './DQL/DQLAgentTraining/{}'.format(filename2)]
    dql_agents = []
    state_size= len(env.repOBS.flatten())
    dql_agents = initialize_agents(arglist, state_size, model_file)
    max_score = float("-inf")
    max_score_timestep = float("inf")


    for episode in range(arglist.num_training):
        print("===============================")
        print("[Deep Q-Learning] @ EPISODE {}".format(episode))
        print("===============================")
        state = np.array(env.reset()).ravel()
        action_history = {agent.name: [] for agent in dql_agents}
        max_score, max_score_timestep = run_episode(env, dql_agents, state, action_history, max_score, max_score_timestep, episode, arglist.rs2)
    
    for agent in dql_agents:
        agent.load_model_trained()
        state = np.array(env.reset()).ravel()
        action_histories = {agent.name: agent.nnmodel.action_history for agent in dql_agents}
        reward_total = run_prediction(env, dql_agents, action_histories,csv_filename, arglist.rs2)
        print("Total Reward:{}".format(reward_total))

            
    
    if not arglist.rs2:
        print("Recipe deliver: ", env.successful)
        print("Lowest Time-step", env.t)
    else:
        print("Highest score: ", reward_total)
        print("Average Time-step", env.t)
    # print("Delivered:", env.delivered)

def run_prediction(env, agents, action_histories,csv_filename, rs2=False):
    reward_total = 0
    done = False
    env.predict = True
    env.reset()
    accumulative_reward = 0
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        while (rs2 and env.t < arglist.max_timestep) or (not rs2 and not done and env.t < arglist.max_timestep):
            action_dict = {agent.name: action_histories[agent.name][env.t] for agent in agents}
            next_state, reward, done, info = env.step(action_dict)
            reward = reward or 0
            if done:
                reward += 50
            next_state = np.array(next_state).ravel()
            reward_total += reward
            if arglist.record_data:
                accumulative_reward += reward
                writer.writerow([env.t, accumulative_reward])

        return reward_total   
def run_episode(env, agents, state, action_history, max_score, max_score_timestep, episode, rs2=False):
    reward_total = 0
    done = False

    while (rs2 and env.t < arglist.max_timestep) or (not rs2 and not done and env.t < arglist.max_timestep):
        action_dict = {}
        for agent in agents:
            agent.agent_actions(env.action_reduction(agent.name))
            action = agent.epsilon_greedy(state)
            if action is False:
                break
            action_dict[agent.name] = action
            action_history[agent.name].append(action)
        next_state, reward, done, _ = env.step(action_dict)
        reward = reward or 0
        next_state = np.array(next_state).ravel()
        [agent.save_transition((state, action_dict, reward, next_state, done)) or agent.update_q_values() or agent.update_target() for agent in agents]
        reward_total += reward
        state = next_state

    if reward_total > max_score or (reward_total == max_score and env.t < max_score_timestep):
        for agent in agents:
            agent.nnmodel.action_history = action_history[agent.name]
            agent.nnmodel.save_model()
        max_score = reward_total
        max_score_timestep = env.t

    print("Episode{} Run Score:{}".format(episode,reward_total))
    return max_score, max_score_timestep
    
    
    



if __name__ == '__main__':
    arglist = parse_arguments()
    
    if arglist.play:
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents,arglist.rs1,arglist.rs2, arglist=arglist)
        game.on_execute()
    elif arglist.dql:
        model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
        assert len(list(filter(lambda x: x is not None,
            model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
        if arglist.rs1:
            assert arglist.model1 is not None, "DQL is not built for rs1, it's only for rs2"
        fix_seed(seed=arglist.seed)
        dqlMainLoop(arglist=arglist)
    else:
        model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
        print("model_types", model_types)
        print("arglist.num_agents", arglist.num_agents)
        assert len(list(filter(lambda x: x is not None,
            model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
        fix_seed(seed=arglist.seed)
        main_loop(arglist=arglist)


