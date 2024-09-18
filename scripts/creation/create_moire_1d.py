from argparse import ArgumentParser
from ase.io import read
from pathlib import Path
from pynmoire.nmoirebuild import (
    NMoireBuild,
    DEFAULT_DESIRED_LATVEC_STRAIN_THRESHOLD,
    DEFAULT_MAX_LATVEC_STRAIN_THRESHOLD,
    DEFAULT_MAX_RMAX,
    DEFAULT_MAX_ITER,
)
import numpy as np
from numpy import int_
from scipy.optimize import Bounds


def list_of_strings(arg):
    return arg.split(",")


def create_moire_1d(
    structures_dir: str,
    structure_files: list,
    structure_file_format: str = "extxyz",
    output_struc_name: str = "moire_output.xyz",
    twist_min_search: float = 0.1,
    twist_max_search: float = 5.0,
    desired_strain: float = DEFAULT_DESIRED_LATVEC_STRAIN_THRESHOLD,
    max_permissible_strain: float = DEFAULT_MAX_LATVEC_STRAIN_THRESHOLD,
    is_1D_dir_1: bool = False,
    is_1D_dir_2: bool = False,
    Rmax_max_search: int = 50,
    max_iter_twist_search: int = 4,
    output_plot_name: str = "best_moire_found.png",
    stiffness_tensors: np.ndarray = None,
):
    """
    Create a moire pattern for 1D structures.

    Parameters:
        structures_dir (str): Path to the directory containing structure files.
        structure_files (list): List of structure file names.
        structure_file_format (str): File format of the structures (default: "extxyz").
        output_struc_name (str): Name of the output structure file (default: "moire_output.xyz").
        twist_min_search (float): Minimum twist angle to search (default: 0.1).
        twist_max_search (float): Maximum twist angle to search (default: 5.0).
        desired_strain (float): Desired strain in lattice vectors (default: DEFAULT_DESIRED_LATVEC_STRAIN_THRESHOLD).
        max_permissible_strain (float): Maximum permissible strain (default: DEFAULT_MAX_LATVEC_STRAIN_THRESHOLD).
        is_1D_dir_1 (bool): Enable 1D along base lattice vector 1 (default: False).
        is_1D_dir_2 (bool): Enable 1D along base lattice vector 2 (default: False).
        Rmax_max_search (int): Maximum super cell size (default: 50).
        max_iter_twist_search (int): Maximum optimization iterations (default: 4).
        output_plot_name (str): Name of the output plot (default: "best_moire_found.png").
        stiffness_tensors (np.ndarray): Stiffness tensors for each structure. If None, default values are used.
    """
    structures_path = Path(structures_dir)
    strucs_to_stack = []
    current_ilayer = 0

    for ct, strucfname in enumerate(structure_files):
        struc = read(
            structures_path / strucfname, "-1", format=structure_file_format
        )
        struc.positions[:, 2] += struc.cell.array[2, 2] / 2
        struc.wrap()
        struc.center(vacuum=20, axis=2)
        struc.arrays["atom_types"] = current_ilayer + np.arange(
            0, struc.get_global_number_of_atoms(), dtype=int_
        )
        current_ilayer = struc.get_global_number_of_atoms()
        strucs_to_stack.append(struc)

    min_twist = np.ones(len(strucs_to_stack) - 1) * twist_min_search
    max_twist = np.ones(len(strucs_to_stack) - 1) * twist_max_search

    if stiffness_tensors is None:
        mose2_stiffness = np.array(
            [[131.34, 32.84, -0.00], [32.88, 131.28, -0.00], [0.00, 0.00, 98.74]]
        )

        mote2_stiffness = np.array(
            [[120.17, 22.89, -0.02], [22.74, 120.15, -0.02], [0.00, 0.00, 97.29]]
        )

        stiffness_tensors = np.vstack((mose2_stiffness[None, :], mote2_stiffness[None, :]))

    valid_twists = Bounds(min_twist, max_twist)
    nmbuilder = NMoireBuild(
        strucs_to_stack,
        stiffness_tensors,
        valid_twists,
        is_1D_dir_1=is_1D_dir_1,
        is_1D_dir_2=is_1D_dir_2,
        max_Rmax_value=Rmax_max_search,
        desired_latvec_strain_threshold=desired_strain,
        max_permitted_latvec_strain_threshold=max_permissible_strain,
    )

    nmbuilder.build(maxiter=max_iter_twist_search, plot_name=output_plot_name)
    nmbuilder.moire_struc.write(output_struc_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--structures-dir",
        required=True,
        help="Absolute/Relative Path of Folder with structures",
    )
    parser.add_argument(
        "--structure-files",
        type=list_of_strings,
        help="Name of structure files in directory",
        required=True,
    )
    parser.add_argument(
        "--structure-file-format",
        help="File format of structure for ASE (cif/extxyz/..). xyz will not work, as it doesn't read lattice vectors",
        default="extxyz",
    )
    parser.add_argument(
        "--output-struc-name",
        help="File format of for output",
        default="moire_output.xyz",
    )
    parser.add_argument(
        "--twist-min-search",
        help="Minimum twist angle to search for lowest elastic energy",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--twist-max-search",
        help="Maximum twist angle to search for lowest elastic energy",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--desired-strain",
        help="Desired strain in lattice vectors will try to reach as close to this as possible",
        type=float,
        default=DEFAULT_DESIRED_LATVEC_STRAIN_THRESHOLD,
    )
    parser.add_argument(
        "--max-permissible-strain",
        help="Maximum strain allowed.",
        type=float,
        default=DEFAULT_MAX_LATVEC_STRAIN_THRESHOLD,
    )
    parser.add_argument(
        "--is-1D-dir-1",
        help="If enabled 1D along base lattice vector 1",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--is-1D-dir-2",
        help="If enabled 1D along base lattice vector 2",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--Rmax-max-search",
        help="Max super cell size",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--max-iter-twist-search",
        help="Max Optimization iterations",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output-plot-name",
        help="Plot Name output",
        type=str,
        default="best_moire_found.png",
    )

    args = parser.parse_args()

    create_moire_1d(
        structures_dir=args.structures_dir,
        structure_files=args.structure_files,
        structure_file_format=args.structure_file_format,
        output_struc_name=args.output_struc_name,
        twist_min_search=args.twist_min_search,
        twist_max_search=args.twist_max_search,
        desired_strain=args.desired_strain,
        max_permissible_strain=args.max_permissible_strain,
        is_1D_dir_1=args.is_1D_dir_1,
        is_1D_dir_2=args.is_1D_dir_2,
        Rmax_max_search=args.Rmax_max_search,
        max_iter_twist_search=args.max_iter_twist_search,
        output_plot_name=args.output_plot_name,
        stiffness_tensors=None,  # You can modify this to accept user input if needed
    )
