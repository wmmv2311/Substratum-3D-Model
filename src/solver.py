# src/solver.py (CORRECTED VERSION)

import numpy as np
from scipy import fftpack
from . import constants as const

def solve_poisson_3d_fft(rho_grid, grid_step):
    """
    Solves the 3D Poisson equation (∇²U = 4πGρ) using the Fast Fourier Transform method.
    """
    print(" -> Solving 3D Poisson equation for gravitational potential...")
    
    nx, ny, nz = rho_grid.shape
    rho_fft = fftpack.fftn(rho_grid)
    
    kx = fftpack.fftfreq(nx, d=grid_step)
    ky = fftpack.fftfreq(ny, d=grid_step)
    kz = fftpack.fftfreq(nz, d=grid_step)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    k_sq = (2 * np.pi)**2 * (KX**2 + KY**2 + KZ**2)
    k_sq[0, 0, 0] = 1.0
    
    potential_fft = - (4 * np.pi * const.G_CONST * rho_fft) / k_sq
    potential_fft[0, 0, 0] = 0
    
    potential_real = np.real(fftpack.ifftn(potential_fft))
    
    print(" -> Gravitational potential U_bar calculated.")
    return potential_real


def solve_phi_field_3d_relax(U_bar, grid_info, tol=1e-5, max_iter=1000, omega=1.8):
    """
    Solves the 3D Klein-Gordon-like equation for the Φ field using the
    Successive Over-Relaxation (SOR) method. CORRECTED IMPLEMENTATION.
    """
    print("Solving 3D equation for Φ field using SOR method...")
    
    grid_step = grid_info['step']
    m_eff_sq = const.M0_SQ_EFF - 2 * const.BETA_SH_UNIVERSAL * U_bar
    m_eff_sq[m_eff_sq < 0] = 1e-6 

    # A better initial guess: a field that decays from the center
    X, Y, Z = grid_info['X'], grid_info['Y'], grid_info['Z']
    r = np.sqrt(X**2 + Y**2 + Z**2)
    phi = np.exp(-r / 5.0)

    h_sq = grid_step**2
    
    for it in range(max_iter):
        phi_old_max = np.max(phi)
        
        # Enforce boundary conditions: phi must be zero at the edges
        phi[0, :, :] = phi[-1, :, :] = 0
        phi[:, 0, :] = phi[:, -1, :] = 0
        phi[:, :, 0] = phi[:, :, -1] = 0

        # Loop over all *interior* points of the grid
        for i in range(1, phi.shape[0] - 1):
            for j in range(1, phi.shape[1] - 1):
                for k in range(1, phi.shape[2] - 1):
                    
                    neighbors_sum = (phi[i+1, j, k] + phi[i-1, j, k] +
                                     phi[i, j+1, k] + phi[i, j-1, k] +
                                     phi[i, j, k+1] + phi[i, j, k-1])
                    
                    phi_old_ijk = phi[i, j, k]
                    
                    term1 = 1.0 / (6.0 + h_sq * m_eff_sq[i, j, k])
                    term2 = neighbors_sum
                    
                    phi_new = (1 - omega) * phi_old_ijk + omega * (term1 * term2)
                    phi[i, j, k] = phi_new

        if it % 20 == 0:
            current_max = np.max(phi)
            error = abs(current_max - phi_old_max)
            print(f"  Iteration: {it}, Max Φ: {current_max:.4f}, Change: {error:.2e}")
            if error < tol and it > 50:
                print(f"Convergence reached at iteration {it}!")
                break
    
    if it == max_iter - 1:
        print("Warning: Maximum iterations reached.")

    print("Φ field calculation finished.")
    return phi
