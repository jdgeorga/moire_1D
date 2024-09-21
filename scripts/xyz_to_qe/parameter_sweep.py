
import os
import json
import itertools
from xyz_to_qe_input import load_config, setup_qe_calculation, prepare_atom

def parameter_sweep(directory, xyz_filename, base_calc_directory, path_to_pseudopotentials, config_file='qe_relax_config.json'):
    # Define parameter ranges
    k_points_list = [[4, 4, 4], [6, 6, 6], [8, 8, 8]]
    ecutwfc_list = [30, 40, 50]
    z_max_list = [20, 26, 32]
    conv_thr_list = [1.0e-10, 1.0e-8, 1.0e-6]  # Added convergence threshold sweep
    
    # Load base config
    base_config = load_config(config_file)
    
    # Create sweep directory
    sweep_dir = os.path.join(base_calc_directory, 'parameter_sweep')
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Iterate over all combinations
    for kpts, ecutwfc, z_max, conv_thr in itertools.product(k_points_list, ecutwfc_list, z_max_list, conv_thr_list):
        # Define a unique subdirectory for this parameter set
        sub_dir = f"kpts_{'_'.join(map(str, kpts))}_ecutwfc_{ecutwfc}_zmax_{z_max}_convthr_{conv_thr:.1e}"
        calc_dir = os.path.join(sweep_dir, sub_dir)
        os.makedirs(calc_dir, exist_ok=True)
        
        # Update config
        config = base_config.copy()
        config['kpts'] = kpts
        config['input_data']['relax']['ecutwfc'] = ecutwfc
        config['input_data']['relax']['electrons']['conv_thr'] = f"{conv_thr:.1e}"  # Update convergence threshold
        
        # Prepare atom with new z_max
        xyz_file = os.path.join(directory, xyz_filename)
        atom = prepare_atom(xyz_file, z_max=z_max)
        
        # Setup QE calculation
        input_file = setup_qe_calculation(atom, calc_dir, 'relax', config)
        print(f"Created input file: {input_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform parameter sweep for Quantum ESPRESSO calculations.")
    parser.add_argument("directory", help="Directory containing the XYZ file")
    parser.add_argument("xyz_filename", help="Name of the XYZ file")
    parser.add_argument("base_calc_directory", help="Base directory for calculation outputs")
    parser.add_argument("path_to_pseudopotentials", help="Path to pseudopotentials")
    parser.add_argument("--config", default="qe_relax_config.json", help="Path to configuration file")
    
    args = parser.parse_args()
    
    parameter_sweep(
        args.directory,
        args.xyz_filename,
        args.base_calc_directory,
        args.path_to_pseudopotentials,
        args.config
    )

if __name__ == "__main__":
    main()
