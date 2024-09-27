import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import argparse
import os
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors



# Set global figure parameters for high quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2


# Add these lines near the top of the file, after the other import statements
n_bins = 256
atom_index_cmap = mcolors.LinearSegmentedColormap.from_list('cyclic', ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], N=n_bins)

def unit_range_fixed(x, L=1, eps=1e-9):
    """
    Wraps the input array x into the range [0, L) with a small epsilon to handle boundary conditions.
    """
    y = x.copy()
    y = y % L
    y[(np.fabs(y) < eps) | (np.fabs(L - y) < eps)] = 0
    return y

def convert_string_to_array(string):
    """
    Converts a string representation of a list to a 2x2 numpy array.
    """
    return np.array([float(x) for x in string.strip('[]').split()]).reshape(2, 2)

def pad_periodic_image(pos, box, n_a1=4, n_a2=20):
    """
    Pads the periodic image of a structure in both the a1 and a2 directions to create a supercell.
    """
    i_range = np.concatenate((np.arange(0, n_a1 + 1), np.arange(-n_a1, 0)))
    j_range = np.concatenate((np.arange(0, n_a2 + 1), np.arange(-n_a2, 0)))

    i, j = np.meshgrid(i_range, j_range)
    i, j = i.flatten(), j.flatten()

    offsets = i[:, np.newaxis] * box[0] + j[:, np.newaxis] * box[1]
    padded_pos = pos[np.newaxis, :, :] + offsets[:, np.newaxis, :]

    return np.vstack((pos, padded_pos[1:].reshape(-1, 2)))

def plot_voronoi_diagram(vor, xlim, ylim, color='k', label=None):
    """
    Creates a Voronoi diagram within specified limits using vectorized operations.
    Returns a LineCollection object instead of plotting directly.
    """
    ridge_vertices = np.array(vor.ridge_vertices)
    valid_ridges = ridge_vertices[(ridge_vertices >= 0).all(axis=1)]
    vertices = vor.vertices[valid_ridges]

    within_limits = np.all((vertices[:, :, 0] >= xlim[0]) & (vertices[:, :, 0] <= xlim[1]) &
                           (vertices[:, :, 1] >= ylim[0]) & (vertices[:, :, 1] <= ylim[1]), axis=1)

    lines = vertices[within_limits]
    lc = LineCollection(lines, colors=color, linewidths=0.5, alpha=1.0, label=label)

    return lc

def get_primitive_voronoi_cell(A1):
    """
    Generates the primitive Voronoi cell for the given lattice vector A1.
    """
    x, y = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    points = np.column_stack((x.ravel(), y.ravel()))
    lattice_points = points @ A1
    vor = Voronoi(lattice_points)
    central_point_index = 4
    central_region = vor.regions[vor.point_region[central_point_index]]
    if -1 not in central_region:
        return vor.vertices[central_region]
    return None

def calculate_voronoi_and_displacement(padded_positions, query_points, A1, n):
    """
    Calculates Voronoi diagrams and displacement vectors.
    """
    points_2d = padded_positions[:, :2]
    vor = Voronoi(points_2d)
    tree = cKDTree(query_points)

    pristine_cell = get_primitive_voronoi_cell(A1)

    first_n_indices = np.arange(n)
    first_n_regions = [vor.regions[vor.point_region[i]] for i in first_n_indices]

    centroids = []
    for region in first_n_regions:
        if -1 not in region:
            cell_vertices = vor.vertices[region]
            centroid = np.mean(cell_vertices, axis=0)
        else:
            centroid = points_2d[first_n_indices[len(centroids)]]
        centroids.append(centroid)

    centroids = np.array(centroids)

    _, nearest_indices = tree.query(centroids)
    closest_query_points = query_points[nearest_indices]

    displacements = closest_query_points - centroids

    confined_displacements = np.array([confine_displacement(disp, vor.vertices[region], pristine_cell) 
                                       for disp, region in zip(displacements, first_n_regions)])

    return vor, centroids, displacements, confined_displacements

def confine_displacement(displacement, actual_cell, pristine_cell):
    """
    Confines the displacement vector within the pristine Voronoi cell using thin plate spline interpolation.
    """
    actual_centroid = np.mean(actual_cell, axis=0)
    pristine_centroid = np.mean(pristine_cell, axis=0)

    actual_cell_centered = actual_cell - actual_centroid
    pristine_cell_centered = pristine_cell - pristine_centroid

    cost_matrix = cdist(actual_cell_centered, pristine_cell_centered)
    _, col_ind = linear_sum_assignment(cost_matrix)

    pristine_cell_matched = pristine_cell[col_ind]

    tps = RBFInterpolator(actual_cell, pristine_cell_matched, kernel='thin_plate_spline', smoothing=0)

    start_point = actual_centroid
    end_point = start_point + displacement

    transformed_points = tps(np.vstack((start_point, end_point)))
    confined_displacement = transformed_points[1] - transformed_points[0]

    return confined_displacement

import matplotlib.colors as mcolors

def visualize_structure(positions, query_points, title, vor, centroids, displacements, cell):
    """
    Visualizes the structure with Voronoi cells, displacement vectors, and unit cell edges.
    """
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)  # Increased width to accommodate colorbar
    query_points_padded = pad_periodic_image(query_points, cell[:2, :2])

    vor_query = Voronoi(query_points_padded)

    mo_voronoi = plot_voronoi_diagram(vor, xlim=(-10, 100), ylim=(-20, 20), color='b')
    w_voronoi = plot_voronoi_diagram(vor_query, xlim=(-10, 100), ylim=(-20, 20), color='r')
    ax.add_collection(mo_voronoi)
    ax.add_collection(w_voronoi)
    ax.legend([mo_voronoi, w_voronoi], ['Mo: Wigner-Seitz Cells', 'W: Wigner-Seitz Cells'])

    ax.scatter(positions[:, 0], positions[:, 1], s=30, c='b', marker='o', label='Mo atoms')
    scatter_w = ax.scatter(query_points[:, 0], query_points[:, 1], c=range(len(query_points)), 
                           cmap=atom_index_cmap, norm=plt.Normalize(0, len(query_points)-1), 
                           s=30, marker='x', label='W atoms')
    
    # Calculate displacement magnitudes
    magnitudes = np.linalg.norm(displacements, axis=1)
    
    # Create a colormap for the arrows
    cmap = plt.cm.Reds
    norm = mcolors.Normalize(vmin=0, vmax=1.8)

    for i, (centroid, displacement, magnitude) in enumerate(zip(centroids, displacements, magnitudes)):
        color = cmap(norm(magnitude))
        ax.arrow(centroid[0], centroid[1], displacement[0], displacement[1], 
                 head_width=0.5, head_length=0.5, fc=color, ec=color, width=0.1, 
                 length_includes_head=True)

    cell_corners = np.array([[0, 0], cell[0, :2], cell[0, :2] + cell[1, :2], cell[1, :2], [0, 0]])
    ax.plot(cell_corners[:, 0], cell_corners[:, 1], 'g-', linewidth=2, label='Unit Cell')

    boundary = 5
    ax.set_xlim(np.min(query_points[:, 0]) - boundary, np.max(query_points[:, 0]) + boundary)
    ax.set_ylim(np.min(query_points[:, 1]) - boundary, np.max(query_points[:, 1]) + boundary)
    ax.set_aspect('equal')
    ax.set_title(f'{title}: Wigner-Seitz Cells of Mo and W Layers', fontsize=16)
    ax.set_xlabel('x (Å)', fontsize=14)
    ax.set_ylabel('y (Å)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='lower right')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Displacement Magnitude (Å)', pad=0.1)
    cbar.ax.tick_params(labelsize=10)
    
    cbar_atom = plt.colorbar(scatter_w, ax=ax, label='W Atom Index', pad=0.02, aspect=30)
    cbar_atom.set_ticks([0, len(query_points)//2, len(query_points)-1])
    cbar_atom.set_ticklabels(['0', f'{len(query_points)//2}', f'{len(query_points)-1}'])


    plt.tight_layout()
    
    os.makedirs('disregistry_plots', exist_ok=True)
    plt.savefig(os.path.join('disregistry_plots', f"{title.lower()}_structure.png"), bbox_inches='tight')
    plt.close(fig)

def visualize_displacements_in_voronoi(confined_displacements, A1, title):
    """
    Visualizes the confined displacement vectors within the pristine Voronoi cell.
    """
    pristine_cell = get_primitive_voronoi_cell(A1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    n_bins = 256
    cm = LinearSegmentedColormap.from_list('cyclic', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'], N=n_bins)

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            offset = i * A1[0] + j * A1[1]
            cell = pristine_cell + offset
            ax.plot(np.append(cell[:, 0], cell[0, 0]),
                    np.append(cell[:, 1], cell[0, 1]), 'k-', linewidth=1, alpha=0.5)

    centroid = np.mean(pristine_cell, axis=0)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            offset = i * A1[0] + j * A1[1]
            end_points = centroid + confined_displacements + offset
            scatter = ax.scatter(end_points[:, 0], end_points[:, 1], 
                                 c=range(len(confined_displacements)), s=50, alpha=0.7, cmap=cm)

    ax.set_xlim(centroid[0] - np.linalg.norm(A1[0]), centroid[0] + np.linalg.norm(A1[0]))
    ax.set_ylim(centroid[1] - np.linalg.norm(A1[1]), centroid[1] + np.linalg.norm(A1[1]))
    ax.set_aspect('equal')
    ax.set_title(f'{title}: Displacements in Pristine Wigner-Seitz Cell', fontsize=16)
    ax.set_xlabel('x (Å)', fontsize=14)
    ax.set_ylabel('y (Å)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    cbar = plt.colorbar(scatter, ax=ax, aspect=30, shrink=0.75)
    cbar.set_label('Atom Index', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(0, len(confined_displacements)-1, 5))
    cbar.set_ticklabels([f'{int(i)}' for i in np.linspace(0, len(confined_displacements)-1, 5)])
    plt.tight_layout()

    os.makedirs('disregistry_plots', exist_ok=True)
    plt.savefig(os.path.join('disregistry_plots', f"{title.lower()}_displacements.png"))
    plt.close(fig)

    return fig, ax, cm, centroid, pristine_cell

def create_displacement_relaxation_gif(displacements_list, A1, title, output_file='displacements_relaxation.gif'):
    """
    Creates an animated GIF showing the relaxation of displacement vectors.
    """
    fig, ax, cm, centroid, pristine_cell = visualize_displacements_in_voronoi(displacements_list[0], A1, title)

    def update_displacement(frame, ax, cm, centroid, pristine_cell, displacements_list, A1):
        print(f"Updating displacement frame {frame + 1}/{len(displacements_list)}")
        ax.clear()

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                offset = i * A1[0] + j * A1[1]
                cell = pristine_cell + offset
                ax.plot(np.append(cell[:, 0], cell[0, 0]),
                        np.append(cell[:, 1], cell[0, 1]), 'k-', linewidth=1, alpha=0.5)

        scatter_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                offset = i * A1[0] + j * A1[1]
                end_points = centroid + displacements_list[frame] + offset
                scatter = ax.scatter(end_points[:, 0], end_points[:, 1], 
                                     c=range(len(end_points)), s=50, alpha=0.7, cmap=cm)
                scatter_list.append(scatter)

        ax.set_xlim(centroid[0] - np.linalg.norm(A1[0]), centroid[0] + np.linalg.norm(A1[0]))
        ax.set_ylim(centroid[1] - np.linalg.norm(A1[1]), centroid[1] + np.linalg.norm(A1[1]))
        ax.set_aspect('equal')
        ax.set_xlabel('x (Å)', fontsize=14)
        ax.set_ylabel('y (Å)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.set_title(f'{title}: Relaxation Step {frame + 1}/{len(displacements_list)}', fontsize=16)

        # Save first and last frame
        if frame == 0 or frame == len(displacements_list) - 1:
            frame_filename = f"{output_file[:-4]}_frame_{frame+1}.png"
            plt.savefig(os.path.join('disregistry_plots', frame_filename), dpi=300, bbox_inches='tight')
            print(f"Saved frame {frame+1} as {frame_filename}")

        return scatter_list

    ani = animation.FuncAnimation(fig, update_displacement, frames=len(displacements_list),
                                  fargs=(ax, cm, centroid, pristine_cell, displacements_list, A1),
                                  interval=200, blit=False)

    os.makedirs('disregistry_plots', exist_ok=True)
    output_path = os.path.join('disregistry_plots', output_file)
    ani.save(output_path, writer='ffmpeg', fps=5)

    plt.close(fig)
    print(f"Displacement animation saved as '{output_path}'")

def create_structure_relaxation_gif(structures_list, A1, title, output_file='structure_relaxation.gif'):
    """
    Creates an animated GIF showing the relaxation of structures with a centered unit cell and periodic repetitions.
    """
    first_structure = structures_list[0]
    positions = first_structure.positions[first_structure.arrays['atom_types'] == 0, :2]
    query_points = first_structure.positions[first_structure.arrays['atom_types'] == 3, :2]
    vor, centroids, displacements, _ = calculate_voronoi_and_displacement(
        pad_periodic_image(positions, first_structure.cell[:2, :2]),
        query_points,
        A1,
        n=len(positions)
    )

    fig = plt.figure(figsize=(12, 12), dpi=300)
    
    ax = fig.add_axes([0.1, 0.15, 0.7, 0.75])

    cell = first_structure.cell[:2, :2]
    cell_center = np.sum(cell, axis=0) / 2
    cell_dimensions = np.linalg.norm(cell, axis=1)
    boundary_x = cell_dimensions[0] * .7
    boundary_y = cell_dimensions[1] * 4

    xlim = (cell_center[0] - cell_dimensions[0]/2 - boundary_x, 
            cell_center[0] + cell_dimensions[0]/2 + boundary_x)
    ylim = (cell_center[1] - cell_dimensions[1]/2 - boundary_y, 
            cell_center[1] + cell_dimensions[1]/2 + boundary_y)

    extension_factor = 3
    xlim_extended = (xlim[0] - (xlim[1] - xlim[0]) * (extension_factor - 1) / 2,
                     xlim[1] + (xlim[1] - xlim[0]) * (extension_factor - 1) / 2)
    ylim_extended = (ylim[0] - (ylim[1] - ylim[0]) * (extension_factor - 1) / 2,
                     ylim[1] + (ylim[1] - ylim[0]) * (extension_factor - 1) / 2)

    norm = plt.Normalize(0, len(query_points)-1)

    cax_atom = fig.add_axes([0.1, 0.3, 0.7, 0.02])
    
    sm = plt.cm.ScalarMappable(cmap=atom_index_cmap, norm=norm)
    sm.set_array([])
    cbar_atom = fig.colorbar(sm, cax=cax_atom, orientation='horizontal', label='Unit Cell W Atom Index')
    cbar_atom.set_ticks(np.linspace(0, len(query_points)-1, 5))
    cbar_atom.set_ticklabels([f'{int(i)}' for i in np.linspace(0, len(query_points)-1, 5)])

    # cmap_displacements = plt.cm.hot_r
    # norm_displacements = mcolors.Normalize(vmin=0, vmax=np.linalg.norm(A1[0, :2] + A1[1, :2]))

    # cax_displacement = fig.add_axes([0.85, 0.375, 0.03, 0.3])
    
    # sm_displacement = plt.cm.ScalarMappable(cmap=cmap_displacements, norm=norm_displacements)
    # sm_displacement.set_array([])
    # cbar_displacement = fig.colorbar(sm_displacement, cax=cax_displacement, label='Displacement Magnitude (Å)')

    def update_structure(frame):
        print(f"Updating structure frame {frame + 1}/{len(structures_list)}")
        
        structure = structures_list[frame]
        positions = structure.positions[structure.arrays['atom_types'] == 0, :2]
        query_points = structure.positions[structure.arrays['atom_types'] == 3, :2]
        vor, centroids, displacements, _ = calculate_voronoi_and_displacement(
            pad_periodic_image(positions, structure.cell[:2, :2]),
            query_points,
            A1,
            n=len(positions)
        )
        
        ax.clear()

        # 1. Create Mo Voronoi LineCollection (lowest z-order)
        mo_voronoi = plot_voronoi_diagram(vor, xlim=xlim_extended, ylim=ylim_extended, color='lightskyblue', label='Mo: Wigner-Seitz Cells')
        ax.add_collection(mo_voronoi)
        mo_voronoi.set_zorder(1)

        # 2. Plot Mo atoms
        ax.scatter(positions[:, 0], positions[:, 1], s=60, c='b', marker='x', label='Mo atoms', zorder=6)

        # 3. Create W Voronoi LineCollection
        padded_query = pad_periodic_image(query_points, structure.cell[:2, :2])
        vor_query = Voronoi(padded_query)
        w_voronoi = plot_voronoi_diagram(vor_query, xlim=xlim_extended, ylim=ylim_extended, color='lightcoral', label='W: Wigner-Seitz Cells')
        ax.add_collection(w_voronoi)
        w_voronoi.set_zorder(3)

        # 4. Plot W atoms
        scatter_w = ax.scatter(query_points[:, 0], query_points[:, 1], 
                            s=60, c=range(len(query_points)), cmap=atom_index_cmap, norm=norm, 
                            marker='o', label='W atoms', zorder=6)

        # 5. Plot displacement arrows
        magnitudes = np.linalg.norm(displacements, axis=1)
        for centroid, displacement, magnitude in zip(centroids, displacements, magnitudes):
            ax.arrow(centroid[0], centroid[1], displacement[0], displacement[1], 
                    head_width=0.3, head_length=0.5, fc='black', ec='black', width=0.1, 
                    length_includes_head=True, zorder=7)

        # Create periodic repetitions
        n_repeats = 2
        for i in range(-n_repeats, n_repeats+1):
            for j in range(-n_repeats*3, n_repeats*3+1):
                offset = i * cell[0] + j * cell[1]
                if i != 0 or j != 0:
                    ax.scatter(positions[:, 0] + offset[0], positions[:, 1] + offset[1], 
                            s=20, c='lightskyblue', marker='x', alpha=1.0, zorder=2)
                    ax.scatter(query_points[:, 0] + offset[0], query_points[:, 1] + offset[1], 
                            s=20, c='lightcoral', marker='o', alpha=1.0, zorder=4)

        # 6. Plot unit cell (highest z-order)
        cell_corners = np.array([[0, 0], cell[0], cell[0] + cell[1], cell[1], [0, 0]])
        ax.plot(cell_corners[:, 0], cell_corners[:, 1], 'g-', linewidth=2, 
                label='Unit Cell', zorder=5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f'{title}: Relaxation Step {frame + 1}/{len(structures_list)}', fontsize=16)
        ax.set_xlabel('x (Å)', fontsize=14)
        ax.set_ylabel('y (Å)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='lower right', fontsize=10)

        # Save first and last frame
        if frame == 0 or frame == len(structures_list) - 1:
            frame_filename = f"{output_file[:-4]}_frame_{frame+1}.png"
            plt.savefig(os.path.join('disregistry_plots', frame_filename), dpi=300, bbox_inches='tight')
            print(f"Saved frame {frame+1} as {frame_filename}")

    ani = animation.FuncAnimation(fig, update_structure, frames=len(structures_list),
                                  interval=200, blit=False)

    output_path = os.path.join('disregistry_plots', output_file)
    ani.save(output_path, writer='ffmpeg', fps=5, dpi=300)

    plt.close(fig)
    print(f"Structure animation saved as '{output_path}'")

def create_displacement_relaxation_gif_with_energy(displacements_list, A1, title, confined_displacements_energy, energies, output_file='displacements_relaxation_with_energy.gif'):
    """
    Creates an animated GIF showing the relaxation of displacement vectors with an energy landscape background.
    """
    pristine_cell = get_primitive_voronoi_cell(A1)
    centroid = np.mean(pristine_cell, axis=0)

    # Create a grid for interpolation
    x = np.linspace(centroid[0] - 1.5*np.linalg.norm(A1[0]), centroid[0] + 1.5*np.linalg.norm(A1[0]), 200)
    y = np.linspace(centroid[1] - 1.5*np.linalg.norm(A1[1]), centroid[1] + 1.5*np.linalg.norm(A1[1]), 200)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    # Repeat confined_displacements and energies
    repeated_displacements = []
    repeated_energies = []
    n_repeats = 2
    for i in range(-n_repeats, n_repeats+1):
        for j in range(-n_repeats, n_repeats+1):
            offset = i * A1[0] + j * A1[1]
            repeated_displacements.append(confined_displacements_energy + offset)
            repeated_energies.append(energies)
    
    repeated_displacements = np.vstack(repeated_displacements)
    repeated_energies = np.concatenate(repeated_energies)

    # Interpolate energies using the repeated data
    rbf = RBFInterpolator(repeated_displacements + centroid, repeated_energies, kernel='thin_plate_spline', smoothing=0.1)
    interpolated_energies = rbf(grid_points).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)  # Increased figure height to accommodate the new colorbar

    norm = plt.Normalize(0, len(displacements_list[0])-1)

    # Plot interpolated energy landscape once
    extent = [x.min(), x.max(), y.min(), y.max()]
    im = ax.imshow(interpolated_energies, extent=extent, origin='lower', cmap='magma', aspect='equal', alpha=1.0)
    
    # Add energy colorbar
    cbar_energy = plt.colorbar(im, ax=ax, label='Energy Relative to Maximum @ AA Stacking (eV)', shrink=0.7, pad=0.1)

    # Create a new axes for the atom index colorbar
    # Adjust the position and width to center it with the imshow plot
    ax_pos = ax.get_position()
    cax_atom = fig.add_axes([ax_pos.x0, ax_pos.y0 - 0.1, ax_pos.width, 0.02])  # [left, bottom, width, height]
    
    sm = plt.cm.ScalarMappable(cmap=atom_index_cmap, norm=norm)
    sm.set_array([])
    cbar_atom = fig.colorbar(sm, cax=cax_atom, orientation='horizontal', label='Unit Cell W Atom Index')
    cbar_atom.set_ticks(np.linspace(0, len(displacements_list[0])-1, 5))
    cbar_atom.set_ticklabels([f'{int(i)}' for i in np.linspace(0, len(displacements_list[0])-1, 5)])

    def update_displacement(frame):
        print(f"Updating displacement frame {frame + 1}/{len(displacements_list)}")
        ax.clear()

        # Replot the energy landscape
        ax.imshow(interpolated_energies, extent=extent, origin='lower', cmap='magma', aspect='equal', alpha=1.0, interpolation='bilinear', zorder=0)
        for i in range(-n_repeats, n_repeats+1):
            for j in range(-n_repeats, n_repeats+1):
                offset = i * A1[0] + j * A1[1]
                cell = pristine_cell + offset
                ax.plot(np.append(cell[:, 0], cell[0, 0]),
                        np.append(cell[:, 1], cell[0, 1]), 'lightskyblue', linewidth=1, alpha=1.0, zorder=1,
                        label='Mo Wigner-Seitz Cells' if i == 0 and j == 0 else None)

                # Add blue 'x' marker for Mo atoms at the center of each Voronoi cell
                cell_center = np.mean(cell, axis=0)
                ax.plot(cell_center[0], cell_center[1], 'bx', markersize=8, markeredgewidth=2, zorder=2)

        scatter_list = []
        for i in range(-n_repeats, n_repeats+1):
            for j in range(-n_repeats, n_repeats+1):
                offset = i * A1[0] + j * A1[1]
                end_points = centroid + displacements_list[frame] + offset
                scatter = ax.scatter(end_points[:, 0], end_points[:, 1], 
                                     c=range(len(end_points)), s=50, alpha=1.0, cmap=atom_index_cmap, norm=norm, zorder=3)
                scatter_list.append(scatter)

        ax.set_xlim(centroid[0] - (4./3.)*np.linalg.norm(A1[0]), centroid[0] + (4./3.)*np.linalg.norm(A1[0]))
        ax.set_ylim(centroid[1] - (4./3.)*np.linalg.norm(A1[1]), centroid[1] + (4./3.)*np.linalg.norm(A1[1]))
        ax.set_aspect('equal')
        ax.set_xlabel('x (Å)', fontsize=14)
        ax.set_ylabel('y (Å)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.set_title(f'{title}: Relaxation Step {frame + 1}/{len(displacements_list)}', fontsize=16)

        # Add legend
        ax.plot([], [], 'bx', markersize=8, markeredgewidth=2, label='Mo atoms')
        ax.scatter([], [], c='r', s=50, alpha=0.7, label='W atoms')
        ax.legend(loc='upper right', fontsize=10)

        # Save first and last frame
        if frame == 0 or frame == len(displacements_list) - 1:
            frame_filename = f"{output_file[:-4]}_frame_{frame+1}.png"
            plt.savefig(os.path.join('disregistry_plots', frame_filename), dpi=300, bbox_inches='tight')
            print(f"Saved frame {frame+1} as {frame_filename}")

        return scatter_list

    ani = animation.FuncAnimation(fig, update_displacement, frames=len(displacements_list),
                                  interval=200, blit=False)

    os.makedirs('disregistry_plots', exist_ok=True)
    output_path = os.path.join('disregistry_plots', output_file)
    ani.save(output_path, writer='ffmpeg', fps=5)

    plt.close(fig)
    print(f"Displacement animation with energy landscape saved as '{output_path}'")
    
def create_relaxation_gifs(unrelaxed_file, relaxed_file, confined_displacements_energy, energies,output_prefix='relaxation'):
    """
    Creates relaxation GIFs for both displacement and structure visualizations.
    """
    unrelaxed = read(unrelaxed_file, index=0, format='extxyz')
    A1 = convert_string_to_array(unrelaxed.info['base_lattice_0'])

    all_structures = read(relaxed_file, index=':', format='extxyz')

    num_steps = 12
    step_indices = np.linspace(0, len(all_structures) - 1, num_steps, dtype=int)
    selected_structures = [all_structures[i] for i in step_indices]

    displacements_list = []
    for structure in selected_structures:
        structure.arrays['atom_types'] = unrelaxed.arrays['atom_types']
        padded_pos = pad_periodic_image(structure.positions[structure.arrays['atom_types'] == 0, :2], structure.cell[:2, :2])
        query_points = structure.positions[structure.arrays['atom_types'] == 3, :2]

        _, _, _, confined_displacements = calculate_voronoi_and_displacement(
            padded_pos,
            query_points,
            A1,
            n=len(structure.positions[structure.arrays['atom_types'] == 0, :2])
        )
        displacements_list.append(confined_displacements)

    # create_displacement_relaxation_gif(displacements_list, A1, 'Disregisty Between Mo (Layer 1) and W (Layer 2)', output_file=f'{output_prefix}_displacements.gif')
    create_structure_relaxation_gif(selected_structures, A1, 'Wigner-Seitz Cells of Mo (Layer 1) and W (Layer 2)', output_file=f'{output_prefix}_structure.gif')
    create_displacement_relaxation_gif_with_energy(displacements_list, A1, 'Disregisty Between Mo (Layer 1) and W (Layer 2)', confined_displacements_energy, energies, output_file='relaxation_displacements_with_energy.gif')

def main(unrelaxed_file, relaxed_file, energy_file, generate_gif=True):
    """
    Main function to run the disregistry analysis.
    """
    print("Starting disregistry analysis...")

    if not os.path.isfile(unrelaxed_file):
        print(f"Error: Unrelaxed file '{unrelaxed_file}' does not exist.")
        return
    if not os.path.isfile(relaxed_file):
        print(f"Error: Relaxed file '{relaxed_file}' does not exist.")
        return
    
    if not os.path.isfile(energy_file):
        print(f"Error: Energy file '{energy_file}' does not exist.")
        return

    print(f"Reading unrelaxed structure from {unrelaxed_file}")
    unrelaxed = read(unrelaxed_file, index=0)
    print(f"Reading relaxed structure from {relaxed_file}")
    relaxed = read(relaxed_file, index=-1)
    relaxed.arrays['atom_types'] = unrelaxed.arrays['atom_types']
    
    print(f"Reading energy file from {energy_file}")
    energy_structures = read(energy_file, index=':', format='extxyz')

    A1 = convert_string_to_array(unrelaxed.info['base_lattice_0'])
    A2 = convert_string_to_array(unrelaxed.info['base_lattice_1'])

    print("Padding positions...")
    padded_relaxed_pos = pad_periodic_image(relaxed.positions[relaxed.arrays['atom_types'] == 0, :2], relaxed.cell[:2, :2])
    padded_unrelaxed_pos = pad_periodic_image(unrelaxed.positions[unrelaxed.arrays['atom_types'] == 0, :2], unrelaxed.cell[:2, :2])
    
    print("Calculating displacements for unrelaxed structure...")
    vor_unrelaxed, centroids_unrelaxed, displacements_unrelaxed, confined_displacements_unrelaxed = calculate_voronoi_and_displacement(
        padded_unrelaxed_pos,
        unrelaxed.positions[unrelaxed.arrays['atom_types'] == 3, :2],
        A1,
        n=len(unrelaxed.positions[unrelaxed.arrays['atom_types'] == 0, :2])
    )

    print("Calculating displacements for relaxed structure...")
    vor_relaxed, centroids_relaxed, displacements_relaxed, confined_displacements_relaxed = calculate_voronoi_and_displacement(
        padded_relaxed_pos,
        relaxed.positions[relaxed.arrays['atom_types'] == 3, :2],
        A1,
        n=len(relaxed.positions[relaxed.arrays['atom_types'] == 0, :2])
    )
    
    print("Calculating displacements for energy structures...")
    vor_energy = []
    centroids_energy = []
    displacements_energy = []
    confined_displacements_energy = []
    energies = []

    for structure in energy_structures:
        padded_energy_pos = pad_periodic_image(structure.positions[structure.arrays['atom_types'] == 0, :2], structure.cell[:2, :2])
        vor, centroids, displacements, confined_displacements = calculate_voronoi_and_displacement(
            padded_energy_pos,
            structure.positions[structure.arrays['atom_types'] == 3, :2],
            A1,
            n=len(structure.positions[structure.arrays['atom_types'] == 0, :2])
        )
        vor_energy.append(vor)
        centroids_energy.append(centroids)
        displacements_energy.append(displacements)
        confined_displacements_energy.append(confined_displacements)
        energies.append(structure.get_potential_energy())
        
    vor_energy = np.array(vor_energy).reshape(-1, 2)
    centroids_energy = np.array(centroids_energy).reshape(-1, 2)
    displacements_energy = np.array(displacements_energy).reshape(-1, 2)
    confined_displacements_energy = np.array(confined_displacements_energy).reshape(-1, 2)
    energies = np.array(energies)
    
    # Shift energies by the maximum energy
    max_energy = np.max(energies)
    energies -= max_energy
    
    print(f"Processed {len(energy_structures)} energy structures.")
    print(f"Shifted energies by maximum energy: {max_energy:.6f} eV")


    print("Visualizing unrelaxed structure...")
    visualize_structure(
        unrelaxed.positions[unrelaxed.arrays['atom_types'] == 0, :2],
        unrelaxed.positions[unrelaxed.arrays['atom_types'] == 3, :2],
        'Unrelaxed',
        vor_unrelaxed,
        centroids_unrelaxed,
        displacements_unrelaxed,
        unrelaxed.cell[:2, :2]
    )

    print("Visualizing relaxed structure...")
    visualize_structure(
        relaxed.positions[relaxed.arrays['atom_types'] == 0, :2],
        relaxed.positions[relaxed.arrays['atom_types'] == 3, :2],
        'Relaxed',
        vor_relaxed,
        centroids_relaxed,
        displacements_relaxed,
        relaxed.cell[:2, :2]
    )

    if generate_gif:
        print("Generating relaxation GIFs...")
        create_relaxation_gifs(unrelaxed_file, relaxed_file, confined_displacements_energy, energies)
    
    print("Disregistry analysis completed.")

if __name__ == "__main__":
    unrelaxed_file = 'MoS2_WSe2_1D.xyz'
    relaxed_file = 'MoS2_WSe2_1D_lammps.traj.xyz'
    energy_file = 'bilayer_MoS2_WSe2_config_min_dist_configuration_space.xyz'
    main(unrelaxed_file, relaxed_file, energy_file, generate_gif=True)