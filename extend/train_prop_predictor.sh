export MASTER_ADDR='localhost'

props_to_train=("plogp" "qed" "sa" "jnk3" "drd2" "gsk3b" "uplogp")
base_port=29501 
gpus_per_job=2 # For example

for i in "${!props_to_train[@]}"; do
  prop="${props_to_train[$i]}"
  current_port=$((base_port + i))
  
  echo "--- Launching training for ${prop} using MASTER_PORT=${current_port} on ${gpus_per_job} GPUs ---"
  
  # No need to export MASTER_PORT here if torchrun sets it, but doesn't hurt.
  # torchrun will set RANK, WORLD_SIZE, LOCAL_RANK for the Python script.
  torchrun --nproc_per_node=$gpus_per_job \
           --master_addr=$MASTER_ADDR \
           --master_port=$current_port \
           -m extend.train_prop_predictor --prop_column_name "$prop" 
done