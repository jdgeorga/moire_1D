import numpy as np
from ase.io import read, write
from ase.constraints import FixedLine
from ase.optimize import FIRE
from moirecompare.calculators import (NLayerCalculator, MonolayerLammpsCalculator, 
                                      InterlayerLammpsCalculator)

def setup_calculators(atoms, layer_symbols):
    intralayer_calcs = []
    interlayer_calcs = []

    for i in range(len(layer_symbols)):
        layer_atoms = atoms[np.logical_and(atoms.arrays['atom_types'] >= i * 3,
                                           atoms.arrays['atom_types'] < (i + 1) * 3)]
        
        intralayer_calcs.append(MonolayerLammpsCalculator(layer_atoms,
                                                          layer_symbols=layer_symbols[i],
                                                          system_type='TMD',
                                                          intra_potential='tmd.sw'))
        
        if i > 0:
            bilayer_atoms = atoms[np.logical_and(atoms.arrays['atom_types'] >= (i - 1) * 3,
                                                 atoms.arrays['atom_types'] < (i + 1) * 3)]
            interlayer_calcs.append(
                InterlayerLammpsCalculator(bilayer_atoms,
                                           layer_symbols=layer_symbols[i - 1:i + 1],
                                           system_type='TMD'))

    return NLayerCalculator(atoms, intralayer_calcs, interlayer_calcs, layer_symbols)

def lammps_relax(input_atoms):
    atoms = input_atoms.copy()
    atoms.cell[2,2] = 100
    layer_symbols = [["Mo", "S", "S"], ["W", "Se", "Se"]]
    
    atoms.calc = setup_calculators(atoms, layer_symbols)

    atoms.calc.calculate(atoms)
    print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
          f"layer_energy {atoms.calc.results['layer_energy']}")
    
    constraint = FixedLine(indices=range(len(atoms)), direction=[0,0,1])
    atoms.set_constraint(constraint)

    dyn = FIRE(atoms)
    dyn.run(fmax=1e-4)

    print(f"Relaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
          f"layer_energy {atoms.calc.results['layer_energy']}")

    return atoms

def lammps_calculate_z_distance(input_atoms, z_distance):
    atoms = input_atoms.copy()
    atoms.cell[2,2] = 100
    layer_symbols = [["Mo", "S", "S"], ["W", "Se", "Se"]]

    # Adjust z-positions to set the specified interlayer distance
    bottom_layer_z = atoms.positions[atoms.arrays['atom_types'] == 0, 2].mean()
    top_layer_z = atoms.positions[atoms.arrays['atom_types'] == 3, 2].mean()
    current_distance = top_layer_z - bottom_layer_z
    z_shift = (z_distance - current_distance) / 2

    atoms.positions[atoms.arrays['atom_types'] < 3, 2] -= z_shift
    atoms.positions[atoms.arrays['atom_types'] >= 3, 2] += z_shift

    atoms.calc = setup_calculators(atoms, layer_symbols)

    atoms.calc.calculate(atoms)
    print(f"Fixed z-distance: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
          f"layer_energy {atoms.calc.results['layer_energy']}")

    return atoms

def determine_minimum_distance(input_atoms, rattle_amplitude=0.01):
    atoms = input_atoms.copy()
    atoms.cell[2,2] = 100
    layer_symbols = [["Mo", "S", "S"], ["W", "Se", "Se"]]
    
    # Apply random displacements (rattling)
    atoms.rattle(stdev=rattle_amplitude, seed=42)
    
    atoms.calc = setup_calculators(atoms, layer_symbols)

    # constraint = FixedLine(indices=range(len(atoms)), direction=[0,0,1])
    # atoms.set_constraint(constraint)

    dyn = FIRE(atoms)
    dyn.run(fmax=1e-5)

    # Calculate the minimum distance between layers
    bottom_layer_z = atoms.positions[atoms.arrays['atom_types'] == 0, 2].mean()
    top_layer_z = atoms.positions[atoms.arrays['atom_types'] == 3, 2].mean()
    min_distance = top_layer_z - bottom_layer_z
    
    write("minimized_atoms.xyz", atoms, format="extxyz")

    print(f"Minimum interlayer distance after relaxation: {min_distance:.3f} Å")
    return min_distance

def generate_configuration_space(bottom_layer_file, top_layer_file, bottom_layer_atom_types, top_layer_atom_types, a1, a2, N, output_prefix, relax=True, z_distance=None, use_min_distance=False):
    # Read bottom and top layer structures
    bottom_layer = read(bottom_layer_file, format="extxyz")
    top_layer = read(top_layer_file, format="extxyz")
    
    # Assign atom types to each layer
    bottom_layer.arrays['atom_types'] = bottom_layer_atom_types
    top_layer.arrays['atom_types'] = top_layer_atom_types
    
    # Adjust z-positions of layers
    bottom_layer.positions[:,2] -= 3
    top_layer.positions[:,2] += 3
    
    # Set z-dimension of cells
    bottom_layer.cell[2,2] = top_layer.cell[2,2] = 20
    
    # Calculate and set mean cell for both layers
    mean_cell = bottom_layer.cell.copy()
    mean_cell[0,:2] = a1
    mean_cell[1,:2] = a2
    
    bottom_layer.set_cell(mean_cell, scale_atoms=True)
    top_layer.set_cell(mean_cell, scale_atoms=True)
    
    # Combine layers into initial structure
    initial_structure = bottom_layer + top_layer
    write("initial_structure.xyz", initial_structure, format="extxyz")

    if use_min_distance:
        z_distance = determine_minimum_distance(initial_structure)

    configurations = []
    for i in range(N):
        for j in range(N):
            displacement = (i/N) * a1 + (j/N) * a2
            current_config = initial_structure.copy()
            top_layer_indices = current_config.arrays['atom_types'] >= 3
            current_config.positions[top_layer_indices,:2] += displacement

            if relax:
                relaxed_atoms = lammps_relax(current_config)
            else:
                relaxed_atoms = lammps_calculate_z_distance(current_config, z_distance)
            
            relaxed_atoms.cell[2,2] = 20
            relaxed_atoms.positions[:,2] += (relaxed_atoms.cell[2,2]/2 - np.mean(relaxed_atoms.positions[:,2]))
            
            relaxed_atoms.info.update({
                'a1_shift_crystal': i/N,
                'a2_shift_crystal': j/N,
                'a1_shift_cartesian': i/N*a1,
                'a2_shift_cartesian': j/N*a2,
                'a1': a1,
                'a2': a2,
                'total_shift': displacement,
                'layer_energy': relaxed_atoms.calc.results['layer_energy'],
                'il_dist': np.abs(np.mean(relaxed_atoms.positions[relaxed_atoms.arrays['atom_types'] == 0, 2] - relaxed_atoms.positions[relaxed_atoms.arrays['atom_types'] == 3, 2]))
            })
            configurations.append(relaxed_atoms)

    write(f"{output_prefix}_configuration_space.xyz", configurations, format="extxyz")
    return configurations

if __name__ == "__main__":
    # Set file paths and parameters
    structures_dir = "/pscratch/sd/j/jdgeorga/twist-anything/1D_moire/pymoire/pymoire/c2db_structures"
    bottom_layer_file = structures_dir + "/MoS2_c2db.xyz"
    top_layer_file = structures_dir + "/WSe2_c2db.xyz"
    
    bottom_layer_atom_types = np.array([0,1,2])
    top_layer_atom_types = np.array([3,4,5])
    periodic_cell = read("MoS2_WSe2_1D.xyz").info['base_lattice_0']
    
    # Helper function to convert string representation of lattice to numpy array
    def convert_string_to_array(string):
        return np.array([float(x) for x in string.strip('[]').split()]).reshape(2,2)
    
    a1, a2 = convert_string_to_array(periodic_cell)
    
    N = 12 # Number of divisions in each direction
    output_prefix = "bilayer_MoS2_WSe2_config_min_dist"

    # Choose between relaxation, fixed z-distance, or minimum distance
    relax = False
    use_min_distance = True
    z_distance = None  # This will be ignored if use_min_distance is True

    # Generate configuration space
    configurations = generate_configuration_space(bottom_layer_file, top_layer_file, bottom_layer_atom_types, top_layer_atom_types, a1, a2, N, output_prefix, relax=relax, z_distance=z_distance, use_min_distance=use_min_distance)