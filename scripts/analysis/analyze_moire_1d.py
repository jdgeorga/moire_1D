import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Remove seaborn import
from scipy.fft import fft
from typing import List, Tuple
from ase import Atoms
from ase.geometry import get_distances
from ase.io import read

def load_atomic_positions(file_path: str) -> List[Atoms]:
    return read(file_path, index=':-1')  # This reads all frames from an XYZ file

def save_heatmap(data: np.ndarray, x_label: str, y_label: str, title: str, filename: str):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='coolwarm', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def save_animated_gif(frames: List[np.ndarray], x_label: str, y_label: str, title: str, filename: str, total_frames: int = 100):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate the step size to achieve the desired number of frames
    step = max(1, len(frames) // total_frames)
    selected_frames = frames[::step]
    
    def update(frame):
        ax.clear()
        im = ax.imshow(frame, cmap='coolwarm', aspect='auto', interpolation='nearest')
        if frame == selected_frames[0]:
            plt.colorbar(im, label='Value')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    
    anim = FuncAnimation(fig, update, frames=selected_frames, interval=200)
    anim.save(filename, writer='pillow')
    plt.close()

def aggregate_atomic_displacement(positions: np.ndarray, initial_positions: np.ndarray, num_bins: int) -> np.ndarray:
    num_steps = positions.shape[0]
    num_atoms = positions.shape[1]
    displacements = positions - initial_positions
    
    bin_edges = np.linspace(0, num_atoms, num_bins + 1, dtype=int)
    heatmap_data = np.zeros((num_steps, num_bins))
    
    for step in range(num_steps):
        for bin in range(num_bins):
            start, end = bin_edges[bin], bin_edges[bin + 1]
            heatmap_data[step, bin] = np.mean(displacements[step, start:end])
            
    print(heatmap_data.shape)
    
    return heatmap_data

def save_line_plot(data: np.ndarray, x_label: str, y_label: str, title: str, filename: str):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data)), data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def visualize_atomic_displacement(positions: np.ndarray, initial_positions: np.ndarray, num_bins: int):
    heatmap_data = aggregate_atomic_displacement(positions, initial_positions, num_bins)
    
    save_heatmap(heatmap_data, 'Spatial Position', 'Relaxation Steps', 
                 'Aggregate Atomic Displacement', 'atomic_displacement_heatmap.png')
    
    save_animated_gif([heatmap_data[:i+1] for i in range(len(heatmap_data))],
                      'Spatial Position', 'Relaxation Steps', 
                      'Aggregate Atomic Displacement', 'atomic_displacement_animation.gif')
    
    # Add line plot
    average_displacement = np.mean(heatmap_data, axis=1)
    save_line_plot(average_displacement, 'Relaxation Steps', 'Average Displacement',
                   'Average Atomic Displacement vs Relaxation Steps', 'atomic_displacement_line_plot.png')

def calculate_strain_evolution(positions: np.ndarray, initial_positions: np.ndarray, num_bins: int) -> np.ndarray:
    num_steps, num_atoms = positions.shape
    initial_distances = np.diff(initial_positions)
    
    strain_data = np.zeros((num_steps, num_atoms - 1))
    for step in range(num_steps):
        distances = np.diff(positions[step])
        strain_data[step] = (distances - initial_distances) / initial_distances
    
    bin_edges = np.linspace(0, num_atoms - 1, num_bins + 1, dtype=int)
    heatmap_data = np.zeros((num_steps, num_bins))
    
    for step in range(num_steps):
        for bin in range(num_bins):
            start, end = bin_edges[bin], bin_edges[bin + 1]
            heatmap_data[step, bin] = np.mean(strain_data[step, start:end])
    
    return heatmap_data

def visualize_strain_evolution(positions: np.ndarray, initial_positions: np.ndarray, num_bins: int):
    heatmap_data = calculate_strain_evolution(positions, initial_positions, num_bins)
    
    save_heatmap(heatmap_data, 'Spatial Position', 'Relaxation Steps', 
                 'Strain Evolution', 'strain_evolution_heatmap.png')
    
    save_animated_gif([heatmap_data[:i+1] for i in range(len(heatmap_data))],
                      'Spatial Position', 'Relaxation Steps', 
                      'Strain Evolution', 'strain_evolution_animation.gif')
    
    # Add line plot
    average_strain = np.mean(heatmap_data, axis=1)
    save_line_plot(average_strain, 'Relaxation Steps', 'Average Strain',
                   'Average Strain vs Relaxation Steps', 'strain_evolution_line_plot.png')

def calculate_potential_energy(atoms_list: List[Atoms]) -> np.ndarray:
    return np.array([atoms.get_potential() for atoms in atoms_list])

def visualize_energy_decay(atoms_list: List[Atoms]):
    energy = calculate_potential_energy(atoms_list)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(energy)), energy)
    plt.xlabel('Relaxation Steps')
    plt.ylabel('Total Potential Energy')
    plt.title('Total Potential Energy Decay')
    plt.savefig('energy_decay.png')
    plt.close()

def calculate_fourier_transform(positions: np.ndarray) -> np.ndarray:
    num_steps, num_atoms = positions.shape
    k = np.fft.fftfreq(num_atoms)
    ft_data = np.zeros((num_steps, num_atoms), dtype=complex)
    
    for step in range(num_steps):
        ft_data[step] = fft(np.exp(-1j * 2 * np.pi * k[:, np.newaxis] * positions[step]))
    
    return np.abs(ft_data)

def visualize_fourier_transform(positions: np.ndarray):
    ft_data = calculate_fourier_transform(positions)
    
    save_heatmap(ft_data, 'Wavevector k', 'Relaxation Steps', 
                 'Dynamic Fourier Transform Evolution', 'fourier_transform_heatmap.png')
    
    save_animated_gif([ft_data[:i+1] for i in range(len(ft_data))],
                      'Wavevector k', 'Relaxation Steps', 
                      'Dynamic Fourier Transform Evolution', 'fourier_transform_animation.gif')

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

def visualize_distance_distribution(positions: np.ndarray, num_bins: int):
    histograms, bin_edges = calculate_distance_distribution(positions, num_bins)
    
    save_heatmap(histograms.T, 'Relaxation Steps', 'Interatomic Distance', 
                 'Interatomic Distance Distribution Evolution', 'distance_distribution_heatmap.png')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        ax.bar(bin_edges[:-1], histograms[frame], width=np.diff(bin_edges), align='edge')
        ax.set_xlabel('Interatomic Distance')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Interatomic Distance Distribution (Step {frame})')
    
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

def visualize_order_parameter_evolution(atoms_list: List[Atoms], output_filename: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_order_parameters = [calculate_order_parameter(atoms) for atoms in atoms_list]
    vmin = min(np.min(op) for op in all_order_parameters)
    vmax = max(np.max(op) for op in all_order_parameters)
    
    def update(frame):
        ax.clear()
        atoms = atoms_list[frame]
        positions = atoms.get_positions()
        order_parameter = all_order_parameters[frame]
        
        scatter = ax.scatter(positions[:, 0], np.zeros_like(positions[:, 0]), 
                             c=order_parameter, cmap='viridis', vmin=vmin, vmax=vmax)
        
        if frame == 0:
            plt.colorbar(scatter, label='Order Parameter (Å)')
        
        ax.set_xlabel('Position (Å)')
        ax.set_yticks([])
        ax.set_title(f'Order Parameter Evolution (Step {frame})')
    
    anim = FuncAnimation(fig, update, frames=len(atoms_list), interval=200)
    anim.save(output_filename, writer='pillow', fps=5)
    plt.close()

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

def visualize_average_order_parameter(atoms_list: List[Atoms], num_bins: int):
    heatmap_data = calculate_average_order_parameter(atoms_list, num_bins)
    
    save_heatmap(heatmap_data, 'Spatial Position', 'Relaxation Steps', 
                 'Average Order Parameter Evolution', 'average_order_parameter_heatmap.png')
    
    # Add line plot
    average_order_parameter = np.mean(heatmap_data, axis=1)
    save_line_plot(average_order_parameter, 'Relaxation Steps', 'Average Order Parameter',
                   'Average Order Parameter vs Relaxation Steps', 'average_order_parameter_line_plot.png')

def main():
    print("Starting analysis...")
    
    # Load atomic positions data as ASE Atoms objects
    print("\n1. Loading atomic positions data...")
    atoms_list = load_atomic_positions('MoS2_WSe2_1D_lammps.traj.xyz')
    print("   Atomic positions loaded successfully.")
    
    # Define parameters
    num_bins = 12
    
    # Run analyses
    positions = np.array([atoms.get_positions() for atoms in atoms_list])
    initial_positions = positions[0]
    
    print("\n2. Analyzing atomic displacement...")
    visualize_atomic_displacement(positions, initial_positions, num_bins)
    print("   Atomic displacement analysis complete.")
    
    print("\n3. Analyzing strain evolution...")
    visualize_strain_evolution(positions, initial_positions, num_bins)
    print("   Strain evolution analysis complete.")
    
    print("\n4. Analyzing energy decay...")
    visualize_energy_decay(atoms_list)
    print("   Energy decay analysis complete.")
    
    print("\n5. Performing Fourier transform analysis...")
    visualize_fourier_transform(positions)
    print("   Fourier transform analysis complete.")
    
    print("\n6. Analyzing distance distribution...")
    visualize_distance_distribution(positions, num_bins)
    print("   Distance distribution analysis complete.")
    
    # Order parameter analyses
    print("\n7. Analyzing order parameter evolution...")
    visualize_order_parameter_evolution(atoms_list, 'order_parameter_evolution.gif')
    print("   Order parameter evolution analysis complete.")
    
    print("\n8. Analyzing average order parameter...")
    visualize_average_order_parameter(atoms_list, num_bins)
    print("   Average order parameter analysis complete.")
    
    print("\nAll analyses completed successfully!")

if __name__ == "__main__":
    main()
