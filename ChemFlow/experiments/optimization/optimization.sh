#!/bin/bash

prop=plogp

## pf
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.optimization --prop $prop --method fp --step_size 0.1 --relative
#id1=$!
#
## limo
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.optimization --prop $prop --method limo --step_size 0.1 --relative
#id2=$!
#
## random
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.optimization --prop $prop --method random --step_size 0.1
#id3=$!
## pde
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.optimization --prop $prop --method wave_sup --step_size 0.1 --relative
#id5=$!
#
#wait $id1 $id2 $id3 $id5
#
## random 1d
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.optimization --prop $prop --method random_1d --step_size 0.1
#id4=$!
## chemspace
#python ChemFlow.experiments/optimization/optimization.py --prop $prop --method chemspace --step_size 0.1
#
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.optimization --prop $prop --method wave_unsup --step_size 0.1 --relative
#id6=$!
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.optimization --prop $prop --method hj_sup --step_size 0.1 --relative
#id7=$!
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.optimization --prop $prop --method hj_unsup --step_size 0.1 --relative
#id8=$!
#
#wait $id4 $id6 $id7 $id8