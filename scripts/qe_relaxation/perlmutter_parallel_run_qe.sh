#!/bin/bash
#SBATCH -J relax_parallel           # Job name
#SBATCH -o relax_parallel.o%j       # Name of stdout output file
#SBATCH -e relax_parallel.e%j       # Name of stderr error file
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH -N 8                         # Total number of nodes
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH -t 04:00:00                  # Run time (hh:mm:ss)
#SBATCH -A m3606                     # Project/Allocation name
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=1

# Load modules
module load espresso/7.0-libxc-5.2.2-gpu
module load cpe/23.03

# Global Variables
TOTAL_NODES=8
TASKS_PER_NODE=4
TOTAL_TASKS=$((TOTAL_NODES * TASKS_PER_NODE))
RUN_DIRECTORIES=(/path/to/dir1 /path/to/dir2 /path/to/dir3 /path/to/dir4) # Add all relevant directories
NUM_RUNS=${#RUN_DIRECTORIES[@]}

# Calculate resources per run
NODES_PER_RUN=$((TOTAL_NODES / NUM_RUNS))
TASKS_PER_RUN=$((TASKS_PER_NODE * NODES_PER_RUN))

# Function to check progress
check_progress() {
    while true; do
        echo "==== Progress Check ===="
        for dir in "${RUN_DIRECTORIES[@]}"; do
            if [ -f "$dir/relax.pwo" ]; then
                last_line=$(tail -n 1 "$dir/relax.pwo")
                # Example parsing, adjust based on actual output format
                relaxation_step=$(echo "$last_line" | grep -oP 'Step\s+\K\d+')
                total_force=$(echo "$last_line" | grep -oP 'Total force = \K[\d.Ee+-]+')
                energy=$(echo "$last_line" | grep -oP 'Energy\s+=\s+\K[\d.Ee+-]+')
                scf_step=$(echo "$last_line" | grep -oP 'SCF step=\s+\K\d+')
                
                echo "Directory: $dir"
                echo "  Relaxation Step: ${relaxation_step:-N/A}"
                echo "  Total Force: ${total_force:-N/A}"
                echo "  Energy Estimation: ${energy:-N/A}"
                echo "  SCF Step Convergence: ${scf_step:-N/A}"
            else
                echo "Directory: $dir - relax.pwo not found."
            fi
        done
        echo "========================"
        sleep 60
    done
}

# Start monitoring in background
check_progress &
MONITOR_PID=$!

# Launch pw.x runs in parallel
for dir in "${RUN_DIRECTORIES[@]}"; do
    echo "Launching pw.x in directory: $dir with $NODES_PER_RUN nodes and $TASKS_PER_RUN tasks."
    (
        cd "$dir" || exit
        srun -N "$NODES_PER_RUN" -n "$TASKS_PER_RUN" --gpus-per-task=1 pw.x -npools 4 -ndiag 1 -in relax.pwi > relax.pwo
    ) &
done

# Wait for all runs to finish
wait

# Kill the monitoring function
kill "$MONITOR_PID"