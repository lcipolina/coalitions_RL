#!/bin/bash
#SBATCH --job-name=ray_example
#SBATCH --account=cstdl
#SBATCH --partition=devel  #batch  devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=39
#SBATCH --time=02:00:00  #aparently max time allowed on devel
#SBATCH --output=ray_job_%j.out


# Load modules or source your Python environment
source /p/home/jusers/cipolina-kun1/juwels/miniconda3/etc/profile.d/conda.sh
conda activate ray_2.6

export RAY_AIR_NEW_OUTPUT=0

# Start the Ray head node
ray start --head --port=6379 --block &

# Sleep for a bit to ensure the head node starts properly
sleep 10

# Run your Python script
python -u /p/home/jusers/cipolina-kun1/juwels/coalitions/ray_init.py

# Stop Ray when done
ray stop
