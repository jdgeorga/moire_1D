import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft
from typing import List, Tuple, Callable
from ase import Atoms
from ase.geometry import get_distances
from ase.io import read
import os
from matplotlib.gridspec import GridSpec


def load_atomic_positions(file_path: str, atom_types_file_path: str) -> List[Atoms]:
    
    traj = read(file_path, index=':-1')  # This reads all frames from an XYZ file
    atom_types_array = read(atom_types_file_path, index='-1', format='extxyz').arrays['atom_types']
    for i in range(len(traj)):
        traj[i].arrays['atom_types'] = atom_types_array
    return traj

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_heatmap(data: np.ndarray, x_label: str, y_label: str, title: str, filename: str):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='coolwarm', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    ensure_directory(os.path.dirname(filename))
    plt.savefig(filename)
    plt.close()

def save_animated_gif(frames: List[np.ndarray], x_label: str, y_label: str, title: str, filename: str, total_frames: int = 10):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate the step size to achieve the desired number of frames
    step = max(1, len(frames) // total_frames)
    selected_frames = frames[::step]
    
    def update(frame):
        ax.clear()
        im = ax.imshow(frame, cmap='coolwarm', aspect='auto', interpolation='nearest')
        if np.array_equal(frame, frames[0]):
            plt.colorbar(im, label='Value')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    
    anim = FuncAnimation(fig, update, frames=selected_frames, interval=200)
    ensure_directory(os.path.dirname(filename))
    anim.save(filename, writer='pillow')
    plt.close()

def save_animated_atom_structure_gif(
    atoms_list: List[Atoms],
    metrics: np.ndarray,
    filename: str,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    atom_types_to_include: List[int],
    total_frames: int = 10
):
    # Calculate the number of rows and columns for subplots
    n_types = len(atom_types_to_include)
    n_cols = min(3, n_types)  # Max 3 columns
    n_rows = (n_types + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), squeeze=False)
    
    # Calculate the step size to achieve the desired number of frames
    step = max(1, len(atoms_list) // total_frames)
    selected_frames = atoms_list[::step]
    selected_metrics = metrics[::step]
    
    # Get initial positions and atom types
    initial_atoms = atoms_list[0]
    initial_positions = initial_atoms.get_positions()
    atom_types = initial_atoms.arrays['atom_types']
    
    # Get cell parameters
    cell = initial_atoms.get_cell()
    
    # Create supercell (5x31)
    nx, ny = 5, 31
    
    # Determine metric range for consistent color scaling
    metric_min = np.min(selected_metrics)
    metric_max = np.max(selected_metrics)
    
    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=metric_min, vmax=metric_max))
    sm.set_array([])
    
    def update(frame_idx):
        for idx, atom_type in enumerate(atom_types_to_include):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.clear()
            
            # Filter for specific atom type
            type_mask = atom_types == atom_type
            filtered_positions = initial_positions[type_mask]
            current_metrics = selected_metrics[frame_idx, type_mask]
            
            supercell_positions = []
            for i in range(-int(np.floor(nx / 2)), int(np.floor(nx / 2)) + 1):
                for j in range(-int(np.floor(ny / 2)), int(np.floor(ny / 2)) + 1):
                    offset = i * cell[0] + j * cell[1]
                    supercell_positions.append(filtered_positions + offset)
            supercell_positions = np.vstack(supercell_positions)
            
            supercell_metrics = np.tile(current_metrics, nx * ny)
            
            # Calculate the range for both x and y
            x_min, x_max = np.min(supercell_positions[:, 1]), np.max(supercell_positions[:, 1])
            y_min, y_max = np.min(supercell_positions[:, 1]), np.max(supercell_positions[:, 1])
            
            # Calculate the maximum range
            max_range = max(x_max - x_min, y_max - y_min)
            
            # Calculate the center points
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # Set square limits
            ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            
            scatter = ax.scatter(
                supercell_positions[:, 0],
                supercell_positions[:, 1],
                c=supercell_metrics,
                cmap='viridis',
                vmin=metric_min,
                vmax=metric_max,
                s=50
            )
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'Atom Type {atom_type}')
            ax.set_aspect('equal')
        
        fig.suptitle(f'{title} (Frame {frame_idx * step})')
    
    anim = FuncAnimation(fig, update, frames=len(selected_frames), interval=200)
    
    # Add colorbar
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), label=colorbar_label)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect parameter as needed
    
    ensure_directory(os.path.dirname(filename))
    anim.save(filename, writer='pillow', fps=5)
    plt.close()

def save_line_plot(data: np.ndarray, x_label: str, y_label: str, title: str, filename: str, atom_types: List[int]):
    plt.figure(figsize=(10, 6))
    
    # Plot line for each atom type
    for i, atom_type in enumerate(atom_types):
        plt.plot(range(len(data)), data[:, i], label=f'Atom Type {atom_type}')
    
    # Plot overall mean
    overall_mean = np.mean(data, axis=1)
    plt.plot(range(len(data)), overall_mean, label='Overall Mean', linewidth=2, color='black')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    ensure_directory(os.path.dirname(filename))
    plt.savefig(filename)
    plt.close()

def get_unique_atom_types(atoms_list: List[Atoms]) -> List[int]:
    return sorted(set(atoms_list[0].arrays['atom_types']))

def visualize_atomic_displacement(atoms_list: List[Atoms], title_prefix: str, focus_atom_types: List[int]):
    print("\n2. Analyzing atomic displacement...")
    
    # Calculate displacements
    positions = np.array([atoms.get_positions() for atoms in atoms_list])
    initial_positions = positions[0]
    
    displacements = np.linalg.norm(positions - initial_positions, axis=2)
    
    # Get atom types
    atom_types = atoms_list[0].arrays['atom_types']
    unique_atom_types = get_unique_atom_types(atoms_list)
    
    # Separate data by atom type
    separated_data = {atom_type: [] for atom_type in unique_atom_types}
    for step in range(len(atoms_list)):
        for atom_type in unique_atom_types:
            type_mask = atom_types == atom_type
            separated_data[atom_type].append(np.mean(displacements[step, type_mask]))
    
    # Convert to numpy array for easier plotting
    heatmap_data = np.array([separated_data[atom_type] for atom_type in unique_atom_types]).T
    
    # Save heatmap
    save_heatmap(heatmap_data, 'Atom Type', 'Relaxation Steps', 
                 f'{title_prefix} - Aggregate Atomic Displacement', 'analysis_plots/atomic_displacement/atomic_displacement_heatmap.png')
    
    # Save animated heatmap
    save_animated_gif([heatmap_data[:i+1] for i in range(len(heatmap_data))],
                      'Atom Type', 'Relaxation Steps', 
                      f'{title_prefix} - Aggregate Atomic Displacement', 'analysis_plots/atomic_displacement/atomic_displacement_animation.gif')
    
    # Save animated atom structure gif
    save_animated_atom_structure_gif(
        atoms_list,
        displacements,
        'analysis_plots/atomic_displacement/atomic_displacement_structure_animation.gif',
        f'{title_prefix} - Atomic Displacement',
        'X position',
        'Y position',
        'Displacement (Å)',
        focus_atom_types
    )
    
    # Calculate and save average displacement line plot
    save_line_plot(heatmap_data, 'Relaxation Steps', 'Average Displacement',
                   f'{title_prefix} - Average Atomic Displacement vs Relaxation Steps', 
                   'analysis_plots/atomic_displacement/atomic_displacement_line_plot.png',
                   unique_atom_types)
    
    print("   Atomic displacement analysis complete.")

def calculate_strain_evolution(positions: np.ndarray, initial_positions: np.ndarray) -> np.ndarray:
    num_steps = positions.shape[0]
    num_atoms = positions.shape[1]
    initial_distances = np.linalg.norm(np.diff(initial_positions, axis=0), axis=1)
    
    strain_data = np.zeros((num_steps, num_atoms - 1))
    for step in range(num_steps):
        distances = np.linalg.norm(np.diff(positions[step], axis=0), axis=1)
        strain_data[step] = (distances - initial_distances) / initial_distances
    
    return strain_data

def visualize_strain_evolution(atoms_list: List[Atoms], title_prefix: str, focus_atom_types: List[int]):
    positions = np.array([atoms.get_positions() for atoms in atoms_list])
    initial_positions = positions[0]
    
    atom_types = atoms_list[0].arrays['atom_types']
    unique_atom_types = get_unique_atom_types(atoms_list)
    
    strain_data = calculate_strain_evolution(positions, initial_positions)
    
    # Separate data by atom type
    separated_data = {atom_type: [] for atom_type in unique_atom_types}
    for step in range(len(atoms_list)):
        for atom_type in unique_atom_types:
            type_mask = atom_types == atom_type
            separated_data[atom_type].append(np.mean(np.abs(strain_data[step, type_mask[:-1]])))  # -1 because strain_data has one less element
    
    # Convert to numpy array for easier plotting
    heatmap_data = np.array([separated_data[atom_type] for atom_type in unique_atom_types]).T
    
    save_heatmap(heatmap_data, 'Atom Type', 'Relaxation Steps', 
                 f'{title_prefix} - Binned Strain Evolution', 'analysis_plots/strain_evolution/binned_strain_evolution_heatmap.png')
    
    save_animated_gif([heatmap_data[:i+1] for i in range(len(heatmap_data))],
                      'Atom Type', 'Relaxation Steps', 
                      f'{title_prefix} - Binned Strain Evolution', 'analysis_plots/strain_evolution/binned_strain_evolution_animation.gif')
    
    # Atom-by-atom strain visualization
    save_heatmap(strain_data.T, 'Relaxation Steps', 'Atom Index', 
                 f'{title_prefix} - Atom-by-Atom Strain Evolution', 'analysis_plots/strain_evolution/atom_by_atom_strain_evolution_heatmap.png')
    
    save_animated_gif([strain_data[:i+1].T for i in range(len(strain_data))],
                      'Relaxation Steps', 'Atom Index', 
                      f'{title_prefix} - Atom-by-Atom Strain Evolution', 'analysis_plots/strain_evolution/atom_by_atom_strain_evolution_animation.gif')
    
    # Animated atom structure with strain coloring
    padded_strain_data = np.pad(strain_data, ((0, 0), (0, 1)), mode='edge')  # Pad to match number of atoms
    save_animated_atom_structure_gif(
        atoms_list,
        metrics=padded_strain_data,
        filename='analysis_plots/strain_evolution/strain_evolution_structure_animation.gif',
        title=f'{title_prefix} - Strain Evolution',
        x_label='X position',
        y_label='Y position',
        colorbar_label='Strain (ε)',
        atom_types_to_include=focus_atom_types,
        total_frames=10
    )
    
    # Line plot of average strain
    save_line_plot(heatmap_data, 'Relaxation Steps', 'Average Strain',
                   f'{title_prefix} - Average Strain vs Relaxation Steps', 
                   'analysis_plots/strain_evolution/strain_evolution_line_plot.png',
                   unique_atom_types)

def calculate_potential_energy(atoms_list: List[Atoms]) -> np.ndarray:
    return np.array([atoms.get_potential_energy() for atoms in atoms_list])

def visualize_energy_decay(atoms_list: List[Atoms], title_prefix: str):
    energy = calculate_potential_energy(atoms_list)
    steps = np.arange(len(energy))

    # Calculate energy relative to final state
    energy_relative = energy - energy[-1]

    # Create a figure with GridSpec for flexible subplot layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Plot 1: Full range, linear scale
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, energy)
    ax1.set_xlabel('Relaxation Steps')
    ax1.set_ylabel('Total Potential Energy')
    ax1.set_title('Full Range (Linear Scale)')

    # Plot 2: Full range, log scale
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(steps, energy_relative)
    ax2.set_xlabel('Relaxation Steps')
    ax2.set_ylabel('Relative Energy (Log Scale)')
    ax2.set_title('Full Range (Log Scale)')

    # Plot 3: Zoomed in, linear scale
    ax3 = fig.add_subplot(gs[1, 0])
    zoom_start = min(len(energy) // 10, 100)  # Start at 10% or 100 steps, whichever is smaller
    ax3.plot(steps[zoom_start:], energy[zoom_start:])
    ax3.set_xlabel('Relaxation Steps')
    ax3.set_ylabel('Total Potential Energy')
    ax3.set_title('Zoomed In (Linear Scale)')

    # Plot 4: Zoomed in, log scale
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(steps[zoom_start:], energy_relative[zoom_start:])
    ax4.set_xlabel('Relaxation Steps')
    ax4.set_ylabel('Relative Energy (Log Scale)')
    ax4.set_title('Zoomed In (Log Scale)')

    plt.tight_layout()
    fig.suptitle(f'{title_prefix} - Total Potential Energy Decay', fontsize=16, y=1.02)
    ensure_directory('analysis_plots/energy_decay')
    plt.savefig('analysis_plots/energy_decay/energy_decay_detailed.png', bbox_inches='tight')
    plt.close()

    print("   Energy decay visualization complete.")

def calculate_fourier_transform(positions: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
    Calculate the Fourier transform of 2D atomic positions relative to the reciprocal lattice vectors.

    Args:
        positions (np.ndarray): Array of shape (num_steps, num_atoms, 2) containing 2D positions.
        a1 (np.ndarray): Lattice vector a1.
        a2 (np.ndarray): Lattice vector a2.

    Returns:
        np.ndarray: Fourier transform magnitudes of shape (num_steps, num_kpoints).
    """
    num_steps, num_atoms, _ = positions.shape

    # Calculate reciprocal lattice vectors
    area = np.abs(a1[0]*a2[1] - a1[1]*a2[0])
    b1 = (2 * np.pi / area) * np.array([a2[1], -a2[0]])
    b2 = (2 * np.pi / area) * np.array([-a1[1], a1[0]])

    # Define k-points in the reciprocal lattice
    k_range = range(-5, 6)  # Adjust the range as needed
    k_points = []
    for i in k_range:
        for j in k_range:
            k = i * b1 + j * b2
            k_points.append(k)
    k_points = np.array(k_points)

    ft_data = np.zeros((num_steps, len(k_points)), dtype=complex)

    for step in range(num_steps):
        for idx, k in enumerate(k_points):
            phase = np.exp(-1j * (positions[step, :, 0] * k[0] + positions[step, :, 1] * k[1]))
            ft_data[step, idx] = np.sum(phase)
    
    return np.abs(ft_data)

def visualize_fourier_transform(
    positions: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    title_prefix: str
):
    """
    Visualize the Fourier transform in reciprocal lattice directions and the Brillouin zone.

    Args:
        positions (np.ndarray): Array of shape (num_steps, num_atoms, 2) containing 2D positions.
        a1 (np.ndarray): Lattice vector a1.
        a2 (np.ndarray): Lattice vector a2.
        title_prefix (str): Prefix for plot titles.
    """
    print("\n4. Revamping Fourier Transform Visualization...")

    # Calculate Fourier transform
    ft_data = calculate_fourier_transform(positions, a1, a2)

    # Define reciprocal lattice vectors
    area = np.abs(a1[0]*a2[1] - a1[1]*a2[0])
    b1 = (2 * np.pi / area) * np.array([a2[1], -a1[0]])
    b2 = (2 * np.pi / area) * np.array([-a1[1], a1[0]])

    # Project Fourier components onto b1 and b2
    ft_b1 = np.dot(ft_data, b1) / len(b1)**2
    ft_b2 = np.dot(ft_data, b2) / len(b2)**2

    steps = np.arange(ft_data.shape[0])

    # Plot Fourier components in b1 and b2 directions
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, ft_b1, label='FT along b1')
    plt.xlabel('Relaxation Step')
    plt.ylabel('Fourier Component')
    plt.title(f'{title_prefix} - FT Components along b1')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(steps, ft_b2, label='FT along b2', color='orange')
    plt.xlabel('Relaxation Step')
    plt.ylabel('Fourier Component')
    plt.title(f'{title_prefix} - FT Components along b2')
    plt.legend()

    plt.tight_layout()
    ensure_directory('analysis_plots/fourier_transform')
    plt.savefig('analysis_plots/fourier_transform/ft_components_b1_b2.png')
    plt.close()

    # Define k-points for Brillouin zone plotting
    num_steps, num_kpoints = ft_data.shape
    k_range = range(-5, 6)  # Ensure consistency with calculate_fourier_transform
    k_points = []
    for i in k_range:
        for j in k_range:
            k = i * b1 + j * b2
            k_points.append(k)
    k_points = np.array(k_points)

    # Normalize Fourier transform for coloring
    ft_normalized = ft_data / np.max(ft_data)

    # Plot Brillouin zone with Fourier transform coloring
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        sc = ax.scatter(
            k_points[:, 0],
            k_points[:, 1],
            c=ft_normalized[frame],
            cmap='viridis',
            s=50,
            edgecolor='k'
        )
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        ax.set_title(f'{title_prefix} - Brillouin Zone FT (Step {frame})')
        plt.colorbar(sc, ax=ax, label='Normalized FT')
        ax.set_aspect('equal')

    anim = FuncAnimation(fig, update, frames=num_steps, interval=200)
    anim.save('analysis_plots/fourier_transform/brillouin_zone_ft_animation.gif', writer='pillow')
    plt.close()

    print("   Fourier transform visualization revamped successfully.")

def calculate_distance_distribution(atoms_list: List[Atoms], num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distribution of interatomic distances for each relaxation step, considering periodic boundaries.

    Args:
        atoms_list (List[Atoms]): List of ASE Atoms objects for each relaxation step.
        num_bins (int): Number of bins for the histogram.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the histograms and the bin edges.
            - histograms: Array of shape (num_steps, num_bins) with counts of distances.
            - bin_edges: Array of bin edge values.
    """
    num_steps = len(atoms_list)
    all_distances = []

    for idx, atoms in enumerate(atoms_list):
        # Obtain all pairwise distances with minimum image convention
        distances = atoms.get_all_distances(mic=True)
        
        # Extract the upper triangle of the distance matrix without the diagonal
        triu_indices = np.triu_indices(len(atoms), k=1)
        unique_distances = distances[triu_indices]
        
        all_distances.append(unique_distances)
        print(f"Step {idx+1}/{num_steps}: Calculated {len(unique_distances)} unique interatomic distances.")

    # Determine global minimum and maximum distances for consistent binning
    all_dists_flat = np.concatenate(all_distances)
    min_dist = all_dists_flat.min()
    max_dist = all_dists_flat.max()
    bin_edges = np.linspace(min_dist, max_dist, num_bins + 1)

    # Initialize histograms array
    histograms = np.zeros((num_steps, num_bins), dtype=int)

    for i, distances in enumerate(all_distances):
        hist, _ = np.histogram(distances, bins=bin_edges)
        histograms[i] = hist
        print(f"Step {i+1}: Histogram computed.")

    return histograms, bin_edges


def visualize_distance_distribution(atoms_list: List[Atoms], num_bins: int, title_prefix: str):
    """
    Visualize the interatomic distance distributions by creating:
    1. A figure with two subplots showing the distribution at the first and last relaxation steps.
    2. A heatmap showing the evolution of distance distributions over relaxation steps.

    Args:
        atoms_list (List[Atoms]): List of ASE Atoms objects for each relaxation step.
        num_bins (int): Number of bins for the histogram.
        title_prefix (str): Prefix for the plot titles.
    """
    print("\nRevamping distance distribution calculation with periodicity...")

    # Calculate distance distributions
    histograms, bin_edges = calculate_distance_distribution(atoms_list, num_bins)
    num_steps = len(atoms_list)

    # 1. Create subplots for the first and last step
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    steps_to_plot = [0, -1]  # First and last steps

    for ax, step in zip(axes, steps_to_plot):
        ax.bar(bin_edges[:-1], histograms[step], width=np.diff(bin_edges), align='edge', color='skyblue', edgecolor='black')
        ax.set_xlabel('Interatomic Distance (Å)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        step_number = step + 1 if step >=0 else num_steps + step +1
        ax.set_title(f'{title_prefix} - Distance Distribution (Step {step_number})', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    subplot_filename = 'analysis_plots/distance_distribution/distance_distribution_subplots.png'
    ensure_directory(os.path.dirname(subplot_filename))
    plt.savefig(subplot_filename)
    plt.close()
    print(f"Saved distance distribution subplots to {subplot_filename}")

    # 2. Create heatmap of distance distributions over relaxation steps
    heatmap_filename = 'analysis_plots/distance_distribution/distance_distribution_heatmap.png'
    save_heatmap(
        data=histograms,
        x_label='Interatomic Distance (Å)',
        y_label='Relaxation Step',
        title=f'{title_prefix} - Interatomic Distance Distribution Evolution',
        filename=heatmap_filename
    )
    print(f"Saved distance distribution heatmap to {heatmap_filename}")

    print("Distance distribution visualization revamped successfully.")

def calculate_order_parameter(atoms: Atoms) -> np.ndarray:
    positions = atoms.get_positions()
    atom_types = atoms.get_atomic_numbers()
    
    type_0_indices = np.where(atom_types == 0)[0]
    type_1_indices = np.where(atom_types == 1)[0]
    
    distances = get_distances(positions[type_0_indices], positions[type_1_indices], 
                              cell=atoms.get_cell(), pbc=atoms.get_pbc())[1]
    
    order_parameter = np.min(distances, axis=1)
    
    return order_parameter

def visualize_order_parameter_evolution(atoms_list: List[Atoms], title_prefix: str):
    print("\n7. Analyzing order parameter evolution...")
    
    # Calculate order parameters for each frame
    all_order_parameters = [calculate_order_parameter(atoms) for atoms in atoms_list]
    order_parameters = np.array(all_order_parameters)  # Shape: (num_frames, num_atoms)
    
    # Save animated GIF using the generalized function
    save_animated_atom_structure_gif(
        atoms_list,
        metrics=order_parameters,
        filename='analysis_plots/order_parameter_evolution/order_parameter_evolution.gif',
        title=f'{title_prefix} - Order Parameter Evolution',
        metric_label='Order Parameter (Å)',
        total_frames=10
    )
    
    print("   Order parameter evolution analysis complete.")

def visualize_average_order_parameter(atoms_list: List[Atoms], title_prefix: str, focus_atom_types: List[int]):
    print("\n7. Analyzing average order parameter...")
    num_steps = len(atoms_list)
    
    order_parameters = np.array([calculate_order_parameter(atoms) for atoms in atoms_list])
    
    atom_types = atoms_list[0].arrays['atom_types']
    unique_atom_types = get_unique_atom_types(atoms_list)
    
    # Separate data by atom type first
    separated_data = {atom_type: [] for atom_type in unique_atom_types}
    for step in range(num_steps):
        for atom_type in unique_atom_types:
            type_mask = atom_types == atom_type
            separated_data[atom_type].append(np.mean(np.abs(order_parameters[step, type_mask])))
    
    # Convert to numpy array for easier plotting
    heatmap_data = np.array([separated_data[atom_type] for atom_type in unique_atom_types]).T
    
    save_heatmap(heatmap_data, 'Atom Type', 'Relaxation Steps', 
                 f'{title_prefix} - Average Order Parameter Evolution', 'analysis_plots/average_order_parameter/average_order_parameter_heatmap.png')
    
    # Use the updated save_line_plot function
    save_line_plot(heatmap_data, 'Relaxation Steps', 'Average Order Parameter Magnitude',
                   f'{title_prefix} - Average Order Parameter Magnitude vs Relaxation Steps', 
                   'analysis_plots/average_order_parameter/average_order_parameter_line_plot.png',
                   unique_atom_types)
    
    print("   Average order parameter analysis complete.")

def run_selected_analyses(atoms_list: List[Atoms], title_prefix: str, focus_atom_types: List[int]):
    analyses = {
        1: ("Atomic displacement", visualize_atomic_displacement),
        2: ("Strain evolution", visualize_strain_evolution),
        3: ("Energy decay", visualize_energy_decay),
        4: ("Fourier transform", visualize_fourier_transform),
        5: ("Distance distribution", lambda al, tp: visualize_distance_distribution(al, 100, tp)),
        6: ("Order parameter evolution", visualize_order_parameter_evolution),
        7: ("Average order parameter", visualize_average_order_parameter)
    }

    print("\nAvailable analyses:")
    for key, (name, _) in analyses.items():
        print(f"{key}. {name}")

    selected = input("\nEnter the numbers of the analyses you want to run (comma-separated, or 'all'): ").strip().lower()

    if selected == 'all':
        selected_keys = list(analyses.keys())
    else:
        selected_keys = [int(k.strip()) for k in selected.split(',') if k.strip().isdigit()]

    # Define lattice vectors a1 and a2
    # These should be defined based on your specific lattice. Example:
    a1 = np.array([3.16, 0.0])  # Replace with actual lattice vector a1
    a2 = np.array([1.58, 2.74])  # Replace with actual lattice vector a2

    for key in selected_keys:
        if key in analyses:
            name, func = analyses[key]
            print(f"\nRunning analysis: {name}")
            if key == 4:
                print("Not Working Yet")
                continue
                positions_2d = np.array([atoms.get_positions()[:, :2] for atoms in atoms_list])
                func(positions_2d, a1, a2, title_prefix)
            elif key in [1, 2, 6, 7]:
                func(atoms_list, title_prefix, focus_atom_types)
            elif key in [3, 5]:
                func(atoms_list, title_prefix)
            print(f"Analysis complete: {name}")
        else:
            print(f"Invalid analysis number: {key}")

def main():
    print("Starting analysis...")
    
    # Load atomic positions data as ASE Atoms objects
    print("\n1. Loading atomic positions data...")
    atoms_list = load_atomic_positions('MoS2_WSe2_1D_lammps.traj.xyz', 'MoS2_WSe2_1D.xyz')
    print("   Atomic positions loaded successfully.")
    
    # Define lattice vectors a1 and a2
    # These should be set according to your system's lattice. Example:
    a1 = np.array([3.16, 0.0])  # Replace with actual lattice vector a1
    a2 = np.array([1.58, 2.74])  # Replace with actual lattice vector a2
    
    # Extract 2D positions
    positions_2d = np.array([atoms.get_positions()[:, :2] for atoms in atoms_list])
    
    # Define parameters
    title_prefix = "MoS2-WSe2 1D Moire"
    focus_atom_types = [0, 3]  # Adjust this list based on your special atom types to focus on
    
    # Create the main analysis_plots directory
    ensure_directory('analysis_plots')
    
    # Run selected analyses
    run_selected_analyses(atoms_list, title_prefix, focus_atom_types)
    
    print("\nAll selected analyses completed successfully!")

if __name__ == "__main__":
    main()
