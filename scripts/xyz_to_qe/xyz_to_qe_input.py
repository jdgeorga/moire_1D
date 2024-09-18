from ase.io import read
from ase.io.espresso import write_espresso_in
import os
import numpy as np
from moirecompare.utils import replace_line_starting_with
from ase.dft.kpoints import bandpath
import json

# Global variable for z_max
Z_MAX = 26

def load_config(config_file='qe_relax_config.json'):
    with open(config_file, 'r') as f:
        return json.load(f)

def setup_qe_calculation(atom, directory, calc_type='relax', config=None):
    if config is None:
        config = load_config(f'qe_{calc_type}_config.json')
    
    input_data = config['input_data'][calc_type]
    pseudopotentials = config['pseudopotentials']
    
    filename = os.path.join(directory, f'{calc_type}.pwi')
    write_espresso_in(filename,
                      atom,
                      input_data=input_data,
                      pseudo_dir=config['path_to_pseudopotentials'],
                      pseudopotentials=pseudopotentials,
                      kpts=config['kpts'])

    # if calc_type == 'relax':
    #     replace_line_starting_with(filename,
    #                                '   conv_thr         = 1e-10',
    #                                '   conv_thr         = 1d-10')

    return filename

def prepare_atom(xyz_file, z_max=Z_MAX):
    atom = read(xyz_file, index=-1, format="extxyz")
    atom.positions[:,2] -= atom.positions[:,2].mean() - z_max / 2
    atom.cell[2,2] = z_max 
    return atom

def main(directory, xyz_filename, calc_directory, path_to_pseudopotentials, config_file='qe_relax_config.json'):
    config = load_config(config_file)
    xyz_file = os.path.join(directory, xyz_filename)
    atom = prepare_atom(xyz_file)

    os.makedirs(calc_directory, exist_ok=True)

    # Add path_to_pseudopotentials to the config
    config['path_to_pseudopotentials'] = path_to_pseudopotentials

    input_file = setup_qe_calculation(atom, calc_directory, 'relax', config)
    print(f"Created input file: {input_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Quantum ESPRESSO input files.")
    parser.add_argument("directory", help="Directory containing the XYZ file")
    parser.add_argument("xyz_filename", help="Name of the XYZ file")
    parser.add_argument("calc_directory", help="Directory for calculation output")
    parser.add_argument("path_to_pseudopotentials", help="Path to pseudopotentials")
    parser.add_argument("--config", default="qe_relax_config.json", help="Path to configuration file")

    args = parser.parse_args()

    main(args.directory, args.xyz_filename, args.calc_directory, args.path_to_pseudopotentials, args.config)