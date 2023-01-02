#!/bin/bash -l
#SBATCH -J first_gpu_job
## Get one node, one CPU-GPU pair
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
## Specify exec time, queue and output
#SBATCH --time=00:10:00
#SBATCH --partition=plgrid-gpu-v100
#SBATCH -A plgpiask2022-gpu
#SBATCH --reservation=piask_tue_gpu
#SBATCH --output=out_heat_gpu_data

## Select module and run task
cd $HOME/labwork/2dheat
./heat_gpu_data_app
