# Overcooked! - Variant 3 - Resource Scarcity

This code is based on and forked from the  "Too many cooks: Bayesian inference for coordinating multi-agent collaboration", Winner of the CogSci 2020 Computational Modeling Prize in High Cognition, and a NeurIPS 2020 CoopAI Workshop Best Paper. from https://github.com/rosewang2008/gym-cooking


Contents:
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Environments and Recipes](docs/environments.md)
- [Design and Customization](docs/design.md)

## Introduction

<p align="center">
    <img src="images/2_open_salad.gif" width=260></img>
    <img src="images/2_partial_tl.gif" width=260></img>
    <img src="images/2_full_salad.gif" width=260></img>
</p>


## Installation

You can install the dependencies with `pip3`:

```
git clone https://github.com/rosewang2008/gym-cooking.git
cd gym-cooking
pip3 install -e .
```

All experiments have been run with `python3`! 

## Usage 

Here, we discuss how to run a single experiment, run our code in manual mode, and re-produce results in our paper. For information on customizing environments, observation/action spaces, and other details, please refer to our section on [Design and Customization](docs/design.md)

For the code below, make sure that you are in **gym-cooking/gym_cooking/**. This means, you should be able to see the file `main.py` in your current directory.

<p align="center">
    <img src="images/2_open.png" width=260></img>
    <img src="images/3_partial.png" width=260></img>
    <img src="images/4_full.png" width=260></img>
</p>

### Running an experiment 

The basic structure of our commands is the following:

`python main.py --num-agents <number> --level <level name> --model1 <model name> --model2 <model name> --model3 <model name> --model4 <model name>`

where `<number>` is the number of agents interacting in the environment (we handle up to 4 agents), `level name` are the names of levels available under the directory `cooking/utils/levels`, omitting the `.txt`.

The `<model name>` are the names of models described in the paper. Specifically `<model name>` can be replaced with:
* `bd` to run Bayesian Delegation,
* `up` for Uniform Priors,
* `dc` for Divide & Conquer,
* `fb` for Fixed Beliefs, and 
* `greedy` for Greedy.

For example, running the salad recipe on the partial divider with 2 agents using Bayesian Delegation looks like:
`python3 main.py --num-agents 2 --level partial-divider_salad --model1 bd --model2 bd`

Or, running the tomato-lettuce recipe on the full divider with 3 agents, one using UP, one with D&C, and the third with Bayesian Delegation:
`python main.py --num-agents 3 --level ResourceScarcity1 --model1 up --model2 dc --model3 bd`

Although our work uses object-oriented representations for observations/states, the `OvercookedEnvironment.step` function returns *image observations* in the `info` object. They can be retrieved with `info['image_obs']`.  

### Additional commands

The above commands can also be appended with the following flags:
* `--record` will save the observation at each time step as an image in `misc/game/record`.

### Manual control

To manually control agents and explore the environment, append the `--play` flag to the above commands. Specifying the model names isn't necessary but the level and the number of agents is still required. For instance, to manually control 2 agents with the salad task on the open divider, run:

`python3 main.py --num-agents 2 --level open-divider_salad --play`
`python main.py --num-agents 2 --level open-divider_salad --play`
`python main.py --num-agents 2 --level ResourceScarcity1 --play`
`python main.py --num-agents 2 --level ResourceScarcity2 --play`
`python main.py --num-agents 2 --level ResourceScarcity3 --play`


This will open up the environment in Pygame. Only one agent can be controlled at a time -- the current active agent can be moved with the arrow keys and toggled by pressing `1`, `2`, `3`, or `4` (up until the actual number of agents of course). Hit the Enter key to save a timestamped image of the current screen to `misc/game/screenshots`.

### Reproducing paper results

To run our full suite of computational experiments (self-play and ad-hoc), we've provided the scrip `run_experiments.sh` that runs our experiments on 20 seeds with `2` agents.

To run on `3` agents, modify `run_experiments.sh` with `num_agents=3`.

### Creating visualizations

To produce the graphs from our paper, navigate to the `gym_cooking/misc/metrics` directory, i.e. 

1. `cd gym_cooking/misc/metrics`.

To generate the timestep and completion graphs, run:

2. `python make_graphs.py --legend --time-step`
3. `python make_graphs.py --legend --completion`

This should generate the results figures that can be found in our paper.

Results for homogenous teams (self-play experiments):
![graphs](images/graphs.png)

Results for heterogeneous teams (ad-hoc experiments):
![heatmaps](images/heatmaps.png)

### Activating Resource Scaricty Version 1
* `--rs1` To activate Resource scarcity Version 1 - This must be called with `--record`

### Activating Resource Scaricty Version 2
* `--rs2` To activate Resource scarcity Version 2 - This must be called with `--record`

### Activating Deep Q Learning Model
* `--dql` To activate Deep Q learning agents 

### To Record Data
* `--record_data` To activate the data recording
* This can be access at gym_cooking\Results
