for pde in wave hj; do
  for prop in plogp; do
    #  for prop in 1err 2iik; do
    python -m extend.train_wave_pde_prop \
      --prop $prop \
      --model.pde_function $pde 
  done
done