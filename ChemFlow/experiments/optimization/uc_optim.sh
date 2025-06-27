#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method random --prop plogp --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method fp --prop plogp --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method limo --prop plogp --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method wave_sup --prop plogp --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method wave_unsup --prop plogp --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method hj_sup --prop plogp --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method hj_unsup --prop plogp --n 100_000 --batch_size 10_000 --relative

CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method random --prop qed --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method fp --prop qed --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method limo --prop qed --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method wave_sup --prop qed --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method wave_unsup --prop qed --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method hj_sup --prop qed --n 100_000 --batch_size 10_000 --relative
CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.optimization.uc_optim --method hj_unsup --prop qed --n 100_000 --batch_size 10_000 --relative