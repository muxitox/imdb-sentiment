#!/bin/bash
#SBATCH --job-name="test_mnist"
#SBATCH --workdir=.
#SBATCH --qos=debug
#SBATCH --ntasks=4
#SBATCH --gres gpu:1
#SBATCH --time=00:00:30

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python train.py $1 1
