# Main script to run the full 3D simulation.
# This script is intended to be run in a Jupyter Notebook environment like Google Colab.
# Each commented section can be a separate cell in the notebook.

# ==================================
# CELL 1: Setup and Imports
# ==================================
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# This part is for running outside a standard package structure
# In Colab, we will clone the repo and import from 'src'
try:
    from src import constants as const
    from src import model_builder as mb
    from src import solver
except ImportError:
    # If run directly, add the parent directory to the path
    import sys
    sys.path.append(os.path.abspath('..'))
    from src import constants as const
    from src import model_builder as mb
    from src import solver


# ==================================
# CELL 2: Google Drive Mount (FOR COLAB ONLY)
# ==================================
# from google.colab import drive
# drive.mount('/content/drive')
#
# # Create a directory to save results
# RESULTS_PATH = '/content/drive/MyDrive/Substratum_3D_Results'
# os.makedirs(RESULTS_PATH, exist_ok=True)


# ==================================
# CELL 3: Build the Baryonic Model
# ==================================
print("--- STEP 1: Building the Baryonic Density Grid ---")
start_time = time.time()

# We will model the galaxy specified in the constants file or choose one
GALAXY_NAME = 'NGC1097'
rho_bar, grid_info = mb.build_baryonic_density(galaxy_name=GALAXY_NAME)

end_time = time.time()
print(f"Baryonic model built in {end_time - start_time:.2f} seconds.")

# --- Optional: Save the density grid ---
# np.save(os.path.join(RESULTS_PATH, 'rho_bar.npy'), rho_bar)


# ==================================
# CELL 4: Calculate the Gravitational Potential (U_bar)
# ==================================
print("\n--- STEP 2: Solving for the Gravitational Potential U_bar ---")
start_time = time.time()

U_bar = solver.solve_poisson_3d_fft(rho_bar, grid_info['step'])

end_time = time.time()
print(f"Gravitational potential U_bar calculated in {end_time - start_time:.2f} seconds.")

# --- Optional: Save the potential grid ---
# np.save(os.path.join(RESULTS_PATH, 'U_bar.npy'), U_bar)


# ==================================
# CELL 5: Solve for the Substratum Field (Φ)
# ==================================
print("\n--- STEP 3: Solving for the Substratum Field Φ ---")
print(f"Using Universal Beta_Sh = {const.BETA_SH_UNIVERSAL}")
print(f"Using effective M0^2 = {const.M0_SQ_EFF} 1/kpc^2")
start_time = time.time()

# For a full run, we might need more iterations
max_iterations = 2000 if const.RESOLUTION == 'full_run' else 500

phi_field = solver.solve_phi_field_3d_relax(U_bar, grid_info['step'], max_iter=max_iterations)

end_time = time.time()
print(f"Φ field solved in {end_time - start_time:.2f} seconds.")

# --- Save the final phi field ---
# np.save(os.path.join(RESULTS_PATH, 'phi_field_final.npy'), phi_field)


# ==================================
# CELL 6: Visualization and Analysis
# ==================================
print("\n--- STEP 4: Visualizing the Results (2D Slices) ---")

# Get the central indices for slicing
nx, ny, nz = phi_field.shape
mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2

# Get the coordinate axes
x_ax, y_ax, z_ax = grid_info['x'], grid_info['y'], grid_info['z']

# Create a 2x2 plot
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'3D Simulation Results for {GALAXY_NAME} ({const.RESOLUTION} resolution)', fontsize=16)

# 1. Baryonic Density (xy-plane, z=0)
im1 = axs[0, 0].pcolormesh(x_ax, y_ax, np.log10(rho_bar[:, :, mid_z].T + 1e-3), cmap='viridis')
axs[0, 0].set_title('Baryonic Density (log scale) - Galactic Plane')
axs[0, 0].set_xlabel('x [kpc]')
axs[0, 0].set_ylabel('y [kpc]')
fig.colorbar(im1, ax=axs[0, 0])

# 2. Gravitational Potential (xy-plane, z=0)
im2 = axs[0, 1].pcolormesh(x_ax, y_ax, U_bar[:, :, mid_z].T, cmap='inferno')
axs[0, 1].set_title('Gravitational Potential U_bar - Galactic Plane')
axs[0, 1].set_xlabel('x [kpc]')
axs[0, 1].set_ylabel('y [kpc]')
fig.colorbar(im2, ax=axs[0, 1])

# 3. Phi Field (xy-plane, z=0)
im3 = axs[1, 0].pcolormesh(x_ax, y_ax, phi_field[:, :, mid_z].T, cmap='cividis')
axs[1, 0].set_title('Φ Field Amplitude - Galactic Plane')
axs[1, 0].set_xlabel('x [kpc]')
axs[1, 0].set_ylabel('y [kpc]')
fig.colorbar(im3, ax=axs[1, 0])

# 4. Phi Field (xz-plane, y=0) - "Edge-on" view
im4 = axs[1, 1].pcolormesh(x_ax, z_ax, phi_field[:, mid_y, :].T, cmap='cividis')
axs[1, 1].set_title('Φ Field Amplitude - Edge-on View')
axs[1, 1].set_xlabel('x [kpc]')
axs[1, 1].set_ylabel('z [kpc]')
fig.colorbar(im4, ax=axs[1, 1])

for ax in axs.flat:
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig(os.path.join(RESULTS_PATH, 'final_results_plot.png'))
plt.show()
