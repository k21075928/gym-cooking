# Overcooked! - Variant 5 - Resource Scarcity

This code is based on and forked from the  "Too many cooks: Bayesian inference for coordinating multi-agent collaboration", Winner of the CogSci 2020 Computational Modeling Prize in High Cognition, and a NeurIPS 2020 CoopAI Workshop Best Paper. from https://github.com/rosewang2008/gym-cooking.

This implements a continous working environment agents to continuously deliver recipes.

## Installation

You can install the dependencies with `pip3`:

```
git clone https://github.com/k21075928/gym-cooking.git
cd gym-cooking
pip3 install -e .
```

All experiments have been run with `python3 or python`! 


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


### Additional commands

The above commands can also be appended with the following flags:
* `--record` will save the observation at each time step as an image in `misc/game/record`.

### Manual control

To manually control agents and explore the environment, append the `--play` flag to the above commands. Specifying the model names isn't necessary but the level and the number of agents is still required. For instance, to manually control 2 agents with the salad task on the open divider, run:

`python main.py --num-agents 2 --level open-divider_salad --play`


This will open up the environment in Pygame. Only one agent can be controlled at a time -- the current active agent can be moved with the arrow keys and toggled by pressing `1`, `2`, `3`, or `4` (up until the actual number of agents of course). Hit the Enter key to save a timestamped image of the current screen to `misc/game/screenshots`.


### Activating Resource Scaricty Version 1
* `--rs1` To activate Resource scarcity Version 1 - This must be called with `--record`

### Activating Resource Scaricty Version 2
* `--rs2` To activate Resource scarcity Version 2 - This must be called with `--record`

### Activating Deep Q Learning Model
* `--dql` To activate Deep Q learning agents 

### To Record Data
* `--record_data` To activate the data recording
* This can be access at gym_cooking\Results
