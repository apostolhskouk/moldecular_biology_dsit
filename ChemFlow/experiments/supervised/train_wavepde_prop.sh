#!/bin/bash
MY_WANDB_ENTITY="lakhs"

for pde in wave hj; do
  for prop in plogp qed sa jnk3 drd2 gsk3b uplogp; do
    #  for prop in 1err 2iik; do
    CUDA_VISIBLE_DEVICES=1 python -m ChemFlow.experiments.supervised.train_wavepde_prop \
      --prop $prop \
      --model.learning_rate 1e-3 \
      --model.pde_function $pde \
      --data.n 100_000 \
      --wandb-entity $MY_WANDB_ENTITY \

  done
done
