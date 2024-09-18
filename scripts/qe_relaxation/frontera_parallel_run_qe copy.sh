#!/bin/bash
#SBATCH -J QE_PARALLEL     # Job name
#SBATCH -o QE_PARALLEL.o%j # Name of stdout output file
#SBATCH -e QE_PARALLEL.e%j # Name of stderr error file
#SBATCH -p development     # Queue (partition) name
#SBATCH -N 40              # Total # of nodes
#SBATCH -n 2240            # Total # of mpi tasks
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH -A YOUR_PROJECT    # Project/Allocation name

# Load required modules
module load phdf5

# Set path to pw.x executable
PW=$HOME/codes/q-e/bin/pw.x

# Global Variables
TOTAL_NODES=40
TASKS_PER_NODE=56
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
        # Backup previous output file if it exists
        if [ -f "relax.pwo" ]; then
            mv "relax.pwo" "relax_$(date +%Y%m%d_%H%M%S).pwo"
        fi
        ibrun -n "$TASKS_PER_RUN" task_affinity $PW -npools 5 < relax.pwi > relax.pwo
    ) &
done

# Wait for all runs to finish
wait

# Kill the monitoring function
kill "$MONITOR_PID"

echo "All jobs completed."

