from create_moire_1d import create_moire_1d
import numpy as np

"""
Run Generator for Moire 1D Structure Creation

This script allows you to specify inputs directly to create a Moire 1D structure.
Modify the parameters below to customize your Moire structure generation.

Parameters:
    structures_dir (str): Path to the directory containing structure files.
    structure_files (list): List of structure file names.
    structure_file_format (str): File format of the structures (default: "extxyz").
    output_struc_name (str): Name of the output structure file.
    twist_min_search (float): Minimum twist angle to search.
    twist_max_search (float): Maximum twist angle to search.
    desired_strain (float): Desired strain in lattice vectors.
    max_permissible_strain (float): Maximum permissible strain.
    is_1D_dir_1 (bool): Enable 1D along base lattice vector 1.
    is_1D_dir_2 (bool): Enable 1D along base lattice vector 2.
    Rmax_max_search (int): Maximum super cell size.
    max_iter_twist_search (int): Maximum optimization iterations.
    output_plot_name (str): Name of the output plot.
    stiffness_tensors (np.ndarray): Stiffness tensors for each structure. If None, default values are used.

Usage:
    1. Modify the parameters below as needed.
    2. Run this script: python run_generator.py
"""

# Input parameters
structures_dir = "/home1/08526/jdgeorga/SCRATCH/moire_1D_relaxation/pymoire/pymoire/c2db_structures"
structure_files = ["MoS2_c2db.xyz","WSe2_c2db.xyz"]
structure_file_format = "extxyz"
output_struc_name = "MoS2_WSe2_1D.xyz"
twist_min_search = 0

twist_max_search = 1e-6
desired_strain = 1e-3
max_permissible_strain = 0.2
is_1D_dir_1 = False
is_1D_dir_2 = True
Rmax_max_search = 15
max_iter_twist_search = 5
output_plot_name = "my_moire_plot.png"

# Optional: Specify custom stiffness tensors
# If you don't want to use custom stiffness tensors, set this to None
# stiffness_tensors = None
# Example of custom stiffness tensors:
stiffness_tensors = np.array([
    [[131.34, 32.84, -0.00], [32.88, 131.28, -0.00], [0.00, 0.00, 98.74]],
    [[120.17, 22.89, -0.02], [22.74, 120.15, -0.02], [0.00, 0.00, 97.29]]
])

if __name__ == "__main__":
    create_moire_1d(
        structures_dir=structures_dir,
        structure_files=structure_files,
        structure_file_format=structure_file_format,
        output_struc_name=output_struc_name,
        twist_min_search=twist_min_search,
        twist_max_search=twist_max_search,
        desired_strain=desired_strain,
        max_permissible_strain=max_permissible_strain,
        is_1D_dir_1=is_1D_dir_1,
        is_1D_dir_2=is_1D_dir_2,
        Rmax_max_search=Rmax_max_search,
        max_iter_twist_search=max_iter_twist_search,
        output_plot_name=output_plot_name,
        stiffness_tensors=stiffness_tensors,
    )

    print("Moire 1D structure generation completed.")