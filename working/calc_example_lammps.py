from ase.io import read, write  # Import read and write functions from ASE for handling atomic structures
import numpy as np  # Import numpy for numerical operations
from moirecompare.calculators import  NLayerCalculator  # Import custom calculators from moirecompare package
from moirecompare.calculators import (MonolayerLammpsCalculator, 
                                      InterlayerLammpsCalculator) # Import custom calculators from moirecompare package
from ase.optimize import FIRE, BFGS  # Import optimization algorithms from ASE

def lammps_relax(input_file, output_file):
    from ase.io import read, write  # Import read and write functions from ASE for handling atomic structures

    # Read the input atomic structure
    atoms = read(input_file, format="extxyz")
    atoms.positions[:,2] += 15  # Shift the z positions of all atoms by 15 Å
    atoms.cell[2,2] = 100  # Set the z-dimension of the cell to 100 Å to avoid interactions through periodic boundaries
    print(atoms.positions[:,2])  # Print the new z positions of atoms
    
    atoms.positions[atoms.arrays["atom_types"] >= 3,1] += 0.1
    # Create copies of atoms object based on atom types for different layers
    at_1 = atoms.copy()[atoms.arrays["atom_types"] < 3]
    at_2 = atoms.copy()[atoms.arrays["atom_types"] >= 3]

    # Define atomic symbols for each layer
    layer_symbols = [["Mo", "S", "S"],
                     ["Mo", "S", "S"]]

    # Initialize a list of intralayer calculators
    intralayer_calcs = [
        MonolayerLammpsCalculator(atoms[atoms.arrays['atom_types'] < 3],
                                  layer_symbols[0],
                                  system_type='TMD',
                                  intra_potential='tmd.sw')
    ]
    # Initialize a list of interlayer calculators
    interlayer_calcs = []

    # Loop through the layers and set up calculators
    for i in np.arange(1, len(layer_symbols)):
        layer_atoms = atoms[
            np.logical_and(atoms.arrays['atom_types'] >= i * 3,
                           atoms.arrays['atom_types'] < (i + 1) * 3)
        ]
        print(np.unique(layer_atoms.arrays['atom_types']))  # Print unique atom types in the current layer
        intralayer_calcs.append(MonolayerLammpsCalculator(layer_atoms,
                                                          layer_symbols=layer_symbols[i],
                                                          system_type='TMD',
                                                          intra_potential='tmd.sw'))

        bilayer_atoms = atoms[np.logical_and(atoms.arrays['atom_types'] >= (i - 1) * 3,
                                             atoms.arrays['atom_types'] < (i + 1) * 3)]
        print(np.unique(bilayer_atoms.arrays['atom_types']))  # Print unique atom types in the bilayer
        print(layer_symbols[i - 1:i + 1])  # Print symbols for the current bilayer

        interlayer_calcs.append(
            InterlayerLammpsCalculator(bilayer_atoms,
                                       layer_symbols=layer_symbols[i - 1:i + 1],
                                       system_type='TMD'))

    # Combine the intra- and interlayer calculators into an NLayerCalculator
    n_layer_calc = NLayerCalculator(atoms,
                                    intralayer_calcs,
                                    interlayer_calcs,
                                    layer_symbols)

    # Assign the combined calculator to the atoms object
    atoms.calc = n_layer_calc

    # Perform an initial calculation to get unrelaxed energy
    atoms.calc.calculate(atoms)
    print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
          f"layer_energy {atoms.calc.results['layer_energy']}")

    # Set up the FIRE optimizer for structural relaxation
    dyn = FIRE(atoms, trajectory=f'{output_file}.traj')
    dyn.run(fmax=5e-5)  # Run until the maximum force is below 1e-3 eV/Å

    # Print the relaxed energy
    print(f"Relaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
          f"layer_energy {atoms.calc.results['layer_energy']}")

    from ase.io.trajectory import Trajectory  # Import Trajectory module from ASE for handling trajectories

    # Read the trajectory file generated during relaxation
    traj_path = f"{output_file}.traj"
    traj = Trajectory(traj_path)
    images = [atom for atom in traj]  # Collect all images from the trajectory

    # Write the final relaxed structure to an output file in extxyz format
    write(f"{output_file}.traj.xyz", images, format="extxyz")

# Example usage of the function
if __name__ == "__main__":
    # Define the input XYZ file and output file prefix
    xyz_file_path = "MoS2_WSe2_1D.xyz"
    out_file = 'MoS2_WSe2_1D_lammps'

    # Call the relaxation function with specified input and output paths
    lammps_relax(xyz_file_path, out_file)