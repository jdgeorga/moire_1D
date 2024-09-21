import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft
from typing import List, Tuple, Callable
from ase import Atoms
from ase.geometry import get_distances
from ase.io import read

def load_atomic_positions(file_path: str, atom_types_file_path: str) -> List[Atoms]:
    
    traj = read(file_path, index=':-1')  # This reads all frames from an XYZ file
    atom_types_array = read(atom_types_file_path, index='-1', format='extxyz').arrays['atom_types']
    for i in range(len(traj)):
        traj[i].arrays['atom_types'] = atom_types_array
    return traj

def save_heatmap(data: np.ndarray, x_label: str, y_label: str, title: str, filename: str):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='coolwarm', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
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
    anim.save(filename, writer='pillow')
    plt.close()

def save_animated_atom_structure_gif(
    atoms_list: List[Atoms],
    metrics: np.ndarray,
    filename: str,
    title: str,
    metric_label: str,
    total_frames: int = 10
):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate the step size to achieve the desired number of frames
    step = max(1, len(atoms_list) // total_frames)
    selected_frames = atoms_list[::step]
    selected_metrics = metrics[::step]
    
    # Get initial positions and atom types
    initial_atoms = atoms_list[0]
    initial_positions = initial_atoms.get_positions()
    atom_types = initial_atoms.arrays['atom_types']
    
    # Filter for specific atom types if needed (optional)
    mask = np.logical_or(atom_types == 0, atom_types == 3)
    filtered_positions = initial_positions[mask]
    
    # Ensure metrics match the mask
    if selected_metrics.shape[1] != len(mask):
        print(f"Warning: Metrics shape {selected_metrics.shape} doesn't match mask length {len(mask)}. Adjusting...")
        selected_metrics = selected_metrics[:, mask]
    
    # Get cell parameters
    cell = initial_atoms.get_cell()
    
    # Create supercell (3x3)
    nx, ny = 3, 3
    supercell_positions = []
    for i in range(-int(np.floor(nx / 2)), int(np.floor(nx / 2)) + 1):
        for j in range(-int(np.floor(ny / 2)), int(np.floor(ny / 2)) + 1):
            offset = i * cell[0] + j * cell[1]
            supercell_positions.append(filtered_positions + offset)
    supercell_positions = np.vstack(supercell_positions)
    
    # Determine metric range for consistent color scaling
    metric_min = np.min(selected_metrics)
    metric_max = np.max(selected_metrics)
    
    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=metric_min, vmax=metric_max))
    sm.set_array([])
    
    def update(frame_idx):
        ax.clear()
        current_metrics = selected_metrics[frame_idx]
        
        # Create supercell metrics
        supercell_metrics = np.tile(current_metrics, nx * ny)
        
        scatter = ax.scatter(
            supercell_positions[:, 0],
            supercell_positions[:, 1],
            c=supercell_metrics,
            cmap='viridis',
            vmin=metric_min,
            vmax=metric_max,
            s=50
        )
        
        ax.set_xlabel('X position (Å)')
        ax.set_ylabel('Y position (Å)')
        ax.set_title(f'{title} (Frame {frame_idx * step})')
        ax.set_aspect('equal')
        ax.set_xlim(
            np.min(filtered_positions[:, 0]) - cell[0, 0] / 2,
            np.max(filtered_positions[:, 0]) + cell[0, 0] / 2
        )
        ax.set_ylim(
            np.min(supercell_positions[:, 1]) - cell[1, 1] / 2,
            np.max(supercell_positions[:, 1]) + cell[1, 1] / 2
        )
    
    anim = FuncAnimation(fig, update, frames=len(selected_frames), interval=200)
    
    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax, label=metric_label)
    
    anim.save(filename, writer='pillow', fps=5)
    plt.close()

def save_line_plot(data: np.ndarray, x_label: str, y_label: str, title: str, filename: str):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data)), data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def visualize_atomic_displacement(atoms_list: List[Atoms], num_bins: int, title_prefix: str):
    print("\n2. Analyzing atomic displacement...")
    
    # Calculate displacements
    positions = np.array([atoms.get_positions() for atoms in atoms_list])
    initial_positions = positions[0]
    displacements = np.linalg.norm(positions - initial_positions[:, np.newaxis, :], axis=2)
    
    # Calculate aggregate displacement for heatmap
    bin_edges = np.linspace(0, len(atoms_list[0]), num_bins + 1, dtype=int)
    heatmap_data = np.zeros((len(atoms_list), num_bins))
    
    for step in range(len(atoms_list)):
        for bin in range(num_bins):
            start, end = bin_edges[bin], bin_edges[bin + 1]
            heatmap_data[step, bin] = np.mean(displacements[step, start:end])
    
    # Save heatmap
    save_heatmap(heatmap_data, 'Spatial Position', 'Relaxation Steps', 
                 f'{title_prefix} - Aggregate Atomic Displacement', 'atomic_displacement_heatmap.png')
    
    # Save animated heatmap
    save_animated_gif([heatmap_data[:i+1] for i in range(len(heatmap_data))],
                      'Spatial Position', 'Relaxation Steps', 
                      f'{title_prefix} - Aggregate Atomic Displacement', 'atomic_displacement_animation.gif')
    
    # Save animated atom structure gif
    save_animated_atom_structure_gif(
        atoms_list,
        metrics=displacements,
        filename='atomic_displacement_structure_animation.gif',
        title=f'{title_prefix} - Atomic Displacement',
        metric_label='Displacement (Å)',
        total_frames=10
    )
    
    # Calculate and save average displacement line plot
    average_displacement = np.mean(np.abs(displacements), axis=1)
    save_line_plot(average_displacement, 'Relaxation Steps', 'Average Displacement',
                   f'{title_prefix} - Average Atomic Displacement vs Relaxation Steps', 'atomic_displacement_line_plot.png')
    
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

def visualize_strain_evolution(atoms_list: List[Atoms], num_bins: int, title_prefix: str):
    positions = np.array([atoms.get_positions() for atoms in atoms_list])
    initial_positions = positions[0]
    strain_data = calculate_strain_evolution(positions, initial_positions)
    
    # Binned heatmap
    bin_edges = np.linspace(0, strain_data.shape[1], num_bins + 1, dtype=int)
    heatmap_data = np.zeros((len(atoms_list), num_bins))
    
    for step in range(len(atoms_list)):
        for bin in range(num_bins):
            start, end = bin_edges[bin], bin_edges[bin + 1]
            heatmap_data[step, bin] = np.mean(strain_data[step, start:end])
    
    save_heatmap(heatmap_data, 'Spatial Position', 'Relaxation Steps', 
                 f'{title_prefix} - Binned Strain Evolution', 'binned_strain_evolution_heatmap.png')
    
    save_animated_gif([heatmap_data[:i+1] for i in range(len(heatmap_data))],
                      'Spatial Position', 'Relaxation Steps', 
                      f'{title_prefix} - Binned Strain Evolution', 'binned_strain_evolution_animation.gif')
    
    # Atom-by-atom strain visualization
    save_heatmap(strain_data.T, 'Relaxation Steps', 'Atom Index', 
                 f'{title_prefix} - Atom-by-Atom Strain Evolution', 'atom_by_atom_strain_evolution_heatmap.png')
    
    save_animated_gif([strain_data[:i+1].T for i in range(len(strain_data))],
                      'Relaxation Steps', 'Atom Index', 
                      f'{title_prefix} - Atom-by-Atom Strain Evolution', 'atom_by_atom_strain_evolution_animation.gif')
    
    # Animated atom structure with strain coloring
    padded_strain_data = np.pad(strain_data, ((0, 0), (0, 1)), mode='edge')  # Pad to match number of atoms
    save_animated_atom_structure_gif(
        atoms_list,
        metrics=padded_strain_data,
        filename='strain_evolution_structure_animation.gif',
        title=f'{title_prefix} - Strain Evolution',
        metric_label='Strain (ε)',
        total_frames=10
    )
    
    # Line plot of average strain
    average_strain = np.mean(np.abs(strain_data), axis=1)
    save_line_plot(average_strain, 'Relaxation Steps', 'Average Strain',
                   f'{title_prefix} - Average Strain vs Relaxation Steps', 'strain_evolution_line_plot.png')

def calculate_potential_energy(atoms_list: List[Atoms]) -> np.ndarray:
    return np.array([atoms.get_potential() for atoms in atoms_list])

def visualize_energy_decay(atoms_list: List[Atoms], title_prefix: str):
    energy = calculate_potential_energy(atoms_list)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(energy)), energy)
    plt.xlabel('Relaxation Steps')
    plt.ylabel('Total Potential Energy')
    plt.title(f'{title_prefix} - Total Potential Energy Decay')
    plt.savefig('energy_decay.png')
    plt.close()

def calculate_fourier_transform(positions: np.ndarray) -> np.ndarray:
    num_steps = positions.shape[0]
    num_atoms = positions.shape[1]
    k = np.fft.fftfreq(num_atoms)
    ft_data = np.zeros((num_steps, num_atoms), dtype=complex)
    
    for step in range(num_steps):
        ft_data[step] = fft(np.exp(-1j * 2 * np.pi * k[:, np.newaxis] * positions[step]))
    
    return np.abs(ft_data)

def visualize_fourier_transform(positions: np.ndarray, title_prefix: str):
    ft_data = calculate_fourier_transform(positions)
    
    save_heatmap(ft_data, 'Wavevector k', 'Relaxation Steps', 
                 f'{title_prefix} - Dynamic Fourier Transform Evolution', 'fourier_transform_heatmap.png')
    
    save_animated_gif([ft_data[:i+1] for i in range(len(ft_data))],
                      'Wavevector k', 'Relaxation Steps', 
                      f'{title_prefix} - Dynamic Fourier Transform Evolution', 'fourier_transform_animation.gif')

def calculate_distance_distribution(positions: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    num_steps, num_atoms = positions.shape
    distances = np.diff(positions, axis=1)
    
    all_distances = distances.flatten()
    hist_range = (np.min(all_distances), np.max(all_distances))
    
    histograms = np.zeros((num_steps, num_bins))
    bin_edges = np.linspace(hist_range[0], hist_range[1], num_bins + 1)
    
    for step in range(num_steps):
        histograms[step], _ = np.histogram(distances[step], bins=bin_edges, density=True)
    
    return histograms, bin_edges

def visualize_distance_distribution(positions: np.ndarray, num_bins: int, title_prefix: str):
    histograms, bin_edges = calculate_distance_distribution(positions, num_bins)
    
    save_heatmap(histograms.T, 'Relaxation Steps', 'Interatomic Distance', 
                 f'{title_prefix} - Interatomic Distance Distribution Evolution', 'distance_distribution_heatmap.png')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        ax.bar(bin_edges[:-1], histograms[frame], width=np.diff(bin_edges), align='edge')
        ax.set_xlabel('Interatomic Distance')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{title_prefix} - Interatomic Distance Distribution (Step {frame})')
    
    anim = FuncAnimation(fig, update, frames=len(histograms), interval=200)
    anim.save('distance_distribution_animation.gif', writer='pillow')
    plt.close()

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
        filename='order_parameter_evolution.gif',
        title=f'{title_prefix} - Order Parameter Evolution',
        metric_label='Order Parameter (Å)',
        total_frames=10
    )
    
    print("   Order parameter evolution analysis complete.")

def calculate_average_order_parameter(atoms_list: List[Atoms], num_bins: int) -> np.ndarray:
    num_steps = len(atoms_list)
    num_atoms = len(atoms_list[0])
    
    order_parameters = np.array([calculate_order_parameter(atoms) for atoms in atoms_list])
    
    bin_edges = np.linspace(0, num_atoms, num_bins + 1, dtype=int)
    heatmap_data = np.zeros((num_steps, num_bins))
    
    for step in range(num_steps):
        for bin in range(num_bins):
            start, end = bin_edges[bin], bin_edges[bin + 1]
            heatmap_data[step, bin] = np.mean(order_parameters[step, start:end])
    
    return heatmap_data

def visualize_average_order_parameter(atoms_list: List[Atoms], num_bins: int, title_prefix: str):
    heatmap_data = calculate_average_order_parameter(atoms_list, num_bins)
    
    save_heatmap(heatmap_data, 'Spatial Position', 'Relaxation Steps', 
                 f'{title_prefix} - Average Order Parameter Evolution', 'average_order_parameter_heatmap.png')
    
    # Add line plot
    average_order_parameter = np.mean(heatmap_data, axis=1)
    save_line_plot(average_order_parameter, 'Relaxation Steps', 'Average Order Parameter',
                   f'{title_prefix} - Average Order Parameter vs Relaxation Steps', 'average_order_parameter_line_plot.png')

def main():
    print("Starting analysis...")
    
    # Load atomic positions data as ASE Atoms objects
    print("\n1. Loading atomic positions data...")
    atoms_list = load_atomic_positions('MoS2_WSe2_1D_lammps.traj.xyz', 'MoS2_WSe2_1D.xyz')
    print("   Atomic positions loaded successfully.")
    
    # Define parameters
    num_bins = 12
    title_prefix = "MoS2-WSe2 1D Moire"
    
    # Run analyses
    # Integrate with updated visualization functions that accept arbitrary metrics
    # Example for strain evolution
    print("\n3. Analyzing strain evolution...")
    visualize_strain_evolution(atoms_list, num_bins, title_prefix)
    print("   Strain evolution analysis complete.")
    
    # Similarly, update other visualization functions as needed
    # e.g., visualize_atomic_displacement, visualize_order_parameter_evolution, etc.
    
    print("\nAll analyses completed successfully!")

if __name__ == "__main__":
    main()
