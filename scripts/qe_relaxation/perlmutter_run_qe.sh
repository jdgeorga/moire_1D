#!/bin/bash
#SBATCH -J relax           # Job name
#SBATCH -o relax.o%j       # Name of stdout output file
#SBATCH -e relax.e%j       # Name of stderr error file
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH -N 8    # Total # of nodes
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH -t 04:00:00        # Run time (hh:mm:ss)
#SBATCH -A m3606       # Project/Allocation name (req'd if you have more than 1)
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=1

#Load modules
# module load phdf5
module load espresso/7.0-libxc-5.2.2-gpu
module load cpe/23.03


# Run the srun command in background
srun -N 8 -n 32 --gpus-per-task 1 pw.x -npools 4 -ndiag 1 -in relax.pwi > relax.pwo


# Wait for all background jobs to finish

wait