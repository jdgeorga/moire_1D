import os
import argparse
from ase.io import read, write

def process_directories(directories, output_dir, file_prefix, file_suffix):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for directory in directories:
        atom_list = []
        
        # Find all the files matching the pattern in the directory
        matching_files = [file for file in os.listdir(directory) 
                          if file.startswith(file_prefix) and file.endswith(file_suffix)]
        
        for matching_file in matching_files:
            file_path = os.path.join(directory, matching_file)
            
            # Read the file using ase.io.read
            atoms = read(file_path, index=":-1", format='espresso-out')
            
            # Append the atoms to the atom_list
            atom_list.extend(atoms)
        
        if atom_list:
            # Write the atom_list to an xyz file in the same directory
            output_file = os.path.join(directory, f"{directory}_qe_traj.xyz")
            write(output_file, atom_list, format='extxyz')
            
            # Write the atom_list to an xyz file in the output directory
            output_file = os.path.join(output_dir, f"{directory}_qe_traj.xyz")
            write(output_file, atom_list, format='extxyz')
        else:
            print(f"No matching files found in {directory}")

def main():
    parser = argparse.ArgumentParser(description="Extract atoms objects from Quantum Espresso output files.")
    parser.add_argument("directories", nargs="+", help="List of directories to process")
    parser.add_argument("--output_dir", default="qe_runs_trajs", help="Output directory for combined trajectories")
    parser.add_argument("--file_prefix", default="relax", help="Prefix of files to process")
    parser.add_argument("--file_suffix", default=".pwo", help="Suffix of files to process")
    
    args = parser.parse_args()
    
    process_directories(args.directories, args.output_dir, args.file_prefix, args.file_suffix)

if __name__ == "__main__":
    main()
