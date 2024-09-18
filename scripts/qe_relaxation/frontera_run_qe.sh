#!/bin/bash
#SBATCH -J QE_JOB          # Job name
#SBATCH -o QE_JOB.o%j      # Name of stdout output file
#SBATCH -e QE_JOB.e%j      # Name of stderr error file
#SBATCH -p development     # Queue (partition) name
#SBATCH -N 40              # Total # of nodes
#SBATCH -n 2240            # Total # of mpi tasks
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH -A YOUR_PROJECT    # Project/Allocation name

# Load required modules
module load phdf5

# Set path to pw.x executable
PW=$HOME/codes/q-e/bin/pw.x

# Set job parameters
TOTAL_PROCESSORS=2240
MAX_CONCURRENT_JOBS=1
PROCESSORS_PER_JOB=$((TOTAL_PROCESSORS / MAX_CONCURRENT_JOBS))

# Set input and output file names
INPUT_FILE="relax.pwi"
OUTPUT_FILE="relax.pwo"

# Backup previous output file if it exists
if [ -f "$OUTPUT_FILE" ]; then
    mv "$OUTPUT_FILE" "${OUTPUT_FILE%.pwo}_$(date +%Y%m%d_%H%M%S).pwo"
fi

# Run the job
echo "Running Quantum ESPRESSO pw.x job..."
ibrun -n $PROCESSORS_PER_JOB task_affinity $PW -npools 5 < "$INPUT_FILE" &> "$OUTPUT_FILE"

echo "Job completed."

