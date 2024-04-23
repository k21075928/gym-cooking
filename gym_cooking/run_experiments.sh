#!/bin/bash

# Define the parameters for the experiment
levels=( "ResourceScarcityDQL2")
models=("bd" "dc" "fb" "up" "greedy")
settings=("--rs2" )

# Define the number of agents and seeds

nagents=2

# Loop over each combination of parameters






echo python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 100
python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 100
sleep 5
echo python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 250
python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 250
sleep 5
echo python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 500
 python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 500
sleep 5

echo python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 1000
python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 1000
sleep 5
echo python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 2500
python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 2500
sleep 5
echo python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 5000
python main.py --num-agents 2 --level ResourceScarcityDQL2  --model1 dql --model2 dql  --rs2 --dql --num_training 5000
sleep 5


# for level in "${levels[@]}"; do
#     for model in "${models[@]}"; do
#         for setting in "${settings[@]}"; do
#             # Run the experiment with the current combination of parameters
#             echo python main.py --num-agents $nagents   --level $level --model1 $model --model2 $model  $setting --record
#             python main.py --num-agents $nagents   --level $level --model1 $model --model2 $model  $setting --record
#             sleep 5
#         done
#     done
# done
# for level in "${levels[@]}"; do
#     for index1 in "${!models[@]}"; do
#         for index2 in $(seq $((index1+1)) ${#models[@]}); do
#             model1=${models[$index1]}
#             model2=${models[$index2]}
#             for setting in "${settings[@]}"; do
#                 echo python main.py --num-agents $nagents  --level $level --model1 $model1 --model2 $model2 $setting --record
#                 python main.py --num-agents $nagents  --level $level --model1 $model1 --model2 $model2 $setting --record
#                 sleep 5
#             done
#         done
#     done
# done


# for level in "${levels1[@]}"; do
#     for index1 in "${!models1[@]}"; do
#         for index2 in $(seq $((index1+1)) ${#models1[@]}); do
#             model1=${models1[$index1]}
#             model2=${models1[$index2]}
#             for setting in "${settings[@]}";do
#                 # Run the experiment with the current combination of parameters
#                 echo python main.py --num-agents $nagents  --level $level --model1 $model1 --model2 $model2 $setting --record
#                 python main.py --num-agents $nagents  --level $level --model1 $model1 --model2 $model2 $setting --record
#                 sleep 5
#             done
#         done
#     done
# done



# for level in "${levels2[@]}"; do
#     for index1 in "${!models2[@]}"; do
#         for index2 in $(seq $((index1+1)) ${#models2[@]}); do
#             model1=${models2[$index1]}
#             model2=${models2[$index2]}
#             for setting in "${settings[@]}";do
#                 # Run the experiment with the current combination of parameters
#                 echo python main.py --num-agents $nagents  --level $level --model1 $model1 --model2 $model2 $setting --record
#                 python main.py --num-agents $nagents  --level $level --model1 $model1 --model2 $model2 $setting --record
#                 sleep 5
#             done
#         done
#     done
# done



# for level in "${levels3[@]}"; do
#     for index1 in "${!models3[@]}"; do
#         for index2 in $(seq $((index1+1)) ${#models3[@]}); do
#             model1=${models3[$index1]}
#             model2=${models3[$index2]}
#             for setting in "${settings[@]}"; do
#                 # Run the experiment with the current combination of parameters
#                 echo python main.py --num-agents $nagents  --level $level --model1 $model1 --model2 $model2 $setting --record
#                 python main.py --num-agents $nagents  --level $level --model1 $model1 --model2 $model2 $setting --record
#                 sleep 5
#             done
#         done
#     done
# done
# echo python main.py --num-agents 2  --level ResourceScarcityFinal2 --model1 bd --model2 fb --rs1 --record
# python  main.py --num-agents 2  --level ResourceScarcityFinal2 --model1 bd --model2 fb --rs1 --record
# sleep 5
