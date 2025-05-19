MY_WANDB_ENTITY="lakhs"
for prop in plogp qed sa jnk3 drd2 gsk3b uplogp; do
#for prop in 1err 2iik; do
  python custom/experiments/supervised/train_prop_predictor.py \
    --data.prop $prop \
    --model.optimizer sgd \
    -e 20 \
    -lb \
    --data.n 11000 \
    --entity $MY_WANDB_ENTITY 
done