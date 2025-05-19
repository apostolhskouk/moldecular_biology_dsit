import os
import torch # type: ignore
import torch.distributed as dist # type: ignore
import hostlist # type: ignore
from .general_utils import Config

global_config = Config()

def setup_for_distributed(is_master):
    """
    Overrides the built-in print function to only allow the master process to print to the console.
    
    Args:
        is_master (bool): Flag indicating if the current process is the master process.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    """
    Checks if the distributed computing is available and initialized.
    
    Returns:
        bool: True if the distributed computing is available and initialized; otherwise, False.
    """

    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    """
    Gets the number of processes in the current distributed group.
    
    Returns:
        int: The number of processes in the current group.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    """
    Gets the rank of the current process in the distributed group.
    
    Returns:
        int: The rank of the current process.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    """
    Saves a checkpoint only from the master process.
    """
    if is_main_process():
        torch.save(*args, **kwargs)

# In /data/hdd1/users/akouk/moldecular_biology_dsit/MolTransformer_repo/MolTransformer/model/utils/distributed_utils.py

def init_distributed_mode():
    rank = 0
    world_size_ = 1
    gpu = 0
    dist_init_is_needed = False

    # Debug: Print initial relevant environment variables at the very start of the function
    # print(f"[DEBUG init_distributed_mode ENTRY] Initial os.environ['RANK']: {os.environ.get('RANK')}")
    # print(f"[DEBUG init_distributed_mode ENTRY] Initial os.environ['WORLD_SIZE']: {os.environ.get('WORLD_SIZE')}")
    # print(f"[DEBUG init_distributed_mode ENTRY] Initial os.environ['MASTER_ADDR']: {os.environ.get('MASTER_ADDR')}")
    # print(f"[DEBUG init_distributed_mode ENTRY] Initial os.environ['MASTER_PORT']: {os.environ.get('MASTER_PORT')}")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # print('Distributed mode: environment variables RANK and WORLD_SIZE detected.')
        rank = int(os.environ["RANK"])
        world_size_ = int(os.environ["WORLD_SIZE"])
        if "LOCAL_RANK" in os.environ:
            gpu = int(os.environ["LOCAL_RANK"])
        elif torch.cuda.is_available():
            gpu = rank % torch.cuda.device_count()
        else:
            gpu = 0
        # print(f'Rank: {rank}, World Size: {world_size_}, GPU: {gpu}')
        dist_init_is_needed = True
    elif all(var in os.environ for var in ["SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"]):
        # print('Distributed mode: SLURM environment variables detected.')
        rank = int(os.environ["SLURM_PROCID"])
        gpu = int(os.environ['SLURM_LOCALID'])
        world_size_ = int(os.environ['SLURM_NTASKS'])
        # print(f'Rank: {rank}, GPU: {gpu}, World Size: {world_size_}')
        dist_init_is_needed = True
    # elif hasattr(global_config, "rank"): # Assuming global_config logic is less relevant here based on error
        # print('Distributed mode: using global_config.rank.') # Simplified for brevity
        # rank = global_config.rank
        # world_size_ = getattr(global_config, "world_size", 1)
        # gpu = getattr(global_config, "local_rank", rank % torch.cuda.device_count() if torch.cuda.is_available() else 0)
        # if world_size_ > 0: dist_init_is_needed = True


    if dist_init_is_needed:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
        
        # Crucial Debug: Check MASTER_PORT value as seen by Python *before* any defaulting.
        python_master_port_env = os.environ.get('MASTER_PORT')
        print(f"--- [PYTHON init_distributed_mode] Shell-provided MASTER_PORT (from os.environ.get): '{python_master_port_env}' ---")

        # Ensure MASTER_ADDR and MASTER_PORT are set in os.environ for init_process_group
        # If the shell script's export worked, os.environ.get('MASTER_PORT') should have the value.
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
            # print(f"Warning: MASTER_ADDR not set. Defaulting to 'localhost'.")
        
        if python_master_port_env is None:
            # This block executes if the shell's MASTER_PORT was NOT received.
            # This is the likely cause of your script trying port 29500.
            default_master_port_fallback = "29500" 
            os.environ['MASTER_PORT'] = default_master_port_fallback
            print(f"--- [PYTHON init_distributed_mode] WARNING: MASTER_PORT not found in Python's os.environ. Defaulting to '{default_master_port_fallback}'. Shell export might not be working. ---")
        else:
            # If python_master_port_env has a value, ensure it's what os.environ['MASTER_PORT'] will be.
            # This is usually redundant if os.environ.get already showed it, but explicit.
            os.environ['MASTER_PORT'] = python_master_port_env 
            print(f"--- [PYTHON init_distributed_mode] Using MASTER_PORT from environment: '{python_master_port_env}' ---")

        # Ensure RANK and WORLD_SIZE from detection are in os.environ for 'env://'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size_)
        
        current_master_addr = os.environ['MASTER_ADDR']
        current_master_port = os.environ['MASTER_PORT'] # This will now be what Python decided based on above logic.
        
        # print(f"Attempting to initialize torch.distributed.init_process_group: "
            #   f"MASTER_ADDR='{current_master_addr}', MASTER_PORT='{current_master_port}', "
            #   f"world_size={world_size_}, rank={rank}")
        
        try:
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=world_size_,
                rank=rank
                # init_method='env://' is default and uses os.environ vars
            )
        except RuntimeError as e:
            if "EADDRINUSE" in str(e) or "address already in use" in str(e):
                print(f"\nCRITICAL EADDRINUSE ERROR on port {current_master_port} for ADDR {current_master_addr}.")
                print("Ensure this port is free or check if multiple processes are unintentionally trying to use the same port due to misconfiguration.")
                print("If shell's MASTER_PORT is not being seen by Python (check debug prints), this is an environment propagation issue.")
            raise

        if world_size_ > 1:
            torch.distributed.barrier()
        # setup_for_distributed(rank == 0) # Make sure setup_for_distributed is defined
    else:
        # print("Not running in distributed mode (defaulting to single GPU/CPU). No process group initialized.")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.set_device(gpu) 
        # setup_for_distributed(True) # Make sure setup_for_distributed is defined

    return world_size_
def reduce_across_processes(val):
    """
    Reduces a value across all processes so that all processes will have the sum of the value.
    
    Args:
        val (number): The value to be reduced.
    
    Returns:
        torch.Tensor: A tensor with the reduced value.
    """
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)
    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t

            


def all_gather_2d(tensor):
    """
    Performs an all-gather operation for a 2D tensor across all processes.
    
    Args:
        tensor (torch.Tensor): The 2D tensor to be gathered.
    
    Returns:
        numpy.ndarray: A numpy array with the gathered tensors from all processes.
    """
    world_size = dist.get_world_size()
    tensor_shape = tensor.shape
    flattened_tensor_shape = (tensor_shape[0] * tensor_shape[1],)
    flattened_tensor = tensor.flatten()
    gathered_tensor_shape = (world_size,) + flattened_tensor_shape
    gathered_tensor = torch.empty(gathered_tensor_shape, dtype=tensor.dtype, device=tensor.device)
    flattened_tensor_list = [torch.empty_like(flattened_tensor) for _ in range(world_size)]
    dist.all_gather(flattened_tensor_list, flattened_tensor)
    for i in range(world_size):
        gathered_tensor[i] = flattened_tensor_list[i]
    output_tensor_shape = (world_size * tensor_shape[0], tensor_shape[1])
    output_tensor = gathered_tensor.view(*output_tensor_shape)
    output_tensor = output_tensor.cpu().detach().numpy()
    return output_tensor


