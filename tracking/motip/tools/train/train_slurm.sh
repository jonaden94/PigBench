#!/bin/bash
#SBATCH -p grete
#SBATCH --nodes=2                # node count
#SBATCH --gpus-per-node=A100:4   # total number of gpus per node
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -C inet
#SBATCH --mem=256G
#SBATCH -A nib00034
#SBATCH -t 0-48:00:00
#SBATCH -o /user/henrich1/u12041/output/job-%J.out

export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

echo "Activating conda..."
source /user/henrich1/u12041/.bashrc
conda activate pigbench
cd ~/repos/PigBench/tracking/motip
monitor_nvml_error

# These environment variables are required for initializing distributed training in pytorch  
export MASTER_PORT=29400
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WANDB_DISABLE_GIT=true

echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$SLURM_NTASKS
echo "MASTER_ADDR="$MASTER_ADDR

srun python main.py --mode train \
                    --config configs/motip.yaml \
                    --exp-name train \
