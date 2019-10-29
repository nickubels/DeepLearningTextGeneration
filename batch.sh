#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=MTGA
#SBATCH --mem=10G
#SBATCH --output=logs/%A_%a-%j.log
#SBATCH --array=1-8%1

INPUTFILE=input.in
ARGS=$(cat $INPUTFILE | head -n $SLURM_ARRAY_TASK_ID | tail -n 1)

module load Python/3.6.4-foss-2018a
module load CUDA/9.1.85
source venv/bin/activate
python3 main.py -j $SLURM_JOB_ID $ARGS
