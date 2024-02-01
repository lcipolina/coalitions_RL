#!/bin/bash
#SBATCH --job-name=ray_cluster
#SBATCH --account=cstdl
#SBATCH --partition=devel
#SBATCH --nodes=2  # Request 2 nodes
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=94
#SBATCH --time=02:00:00
#SBATCH --output=ray_job_%j.out

# Load modules or source your Python environment
source /p/scratch/laionize/cache-kun1/miniconda3/etc/profile.d/conda.sh
conda activate ray_2.6

# Get IP of the first node (head node)
head_node_ip=$(hostname -I | awk '{print $1}')
export ip_head=$head_node_ip:6379

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=6379 --block &

# Sleep to ensure head node starts properly
sleep 10

# Get an array of nodes
nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
worker_num=$((SLURM_JOB_NUM_NODES - 1))

# Start Ray worker nodes
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
    sleep 5
done

# Run your Python script
python -u /p/home/jusers/cipolina-kun1/juwels/coalitions/ray_init.py

# Stop Ray when done
ray stop
