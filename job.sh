#!/bin/bash
#SBATCH --partition=geo
#SBATCH --time=5-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
module load mpich/3.2.1-gnu
mpirun -np 4 python3 rf_log_classifier_joint_3D.py