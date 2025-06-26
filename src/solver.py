import numpy as np
from scipy import fftpack
from . import constants as const

def solve_poisson_3d_fft(rho_grid, grid_step):
    """
    Solves the 3D Poisson equation (∇²U = 4πGρ) using the Fast Fourier Transform method.
    
    Args:
        rho_grid (np.array): A 3D numpy array representing the mass density.
        grid_step (float): The distance between grid points (assumed equal in all dims).
        
    Returns:
        np.array: A 3D numpy array representing the gravitational potential U.
    """
    print(" -> Solving 3D Poisson equation for gravitational potential...")
    
    # 1. Get the shape of the grid
    nx, ny, nz = rho_grid.shape
    
    # 2. Perform the 3D Fast Fourier Transform on the density grid
    rho_fft = fftpack.fftn(rho_grid)
    
    # 3. Create the wave-number grid (k-space)
    kx = fftpack.fftfreq(nx, d=grid_step)
    ky = fftpack.fftfreq(ny, d=grid_step)
    kz = fftpack.fftfreq(nz, d=grid_step)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # 4. Calculate the square of the wave number k² = kx² + ky² + kz²
    # Multiply by (2π)² as required by the Fourier derivative definition
    k_sq = (2 * np.pi)**2 * (KX**2 + KY**2 + KZ**2)
    
    # 5. Avoid division by zero for the k=0 (DC) component
    k_sq[0, 0, 0] = 1.0 
    
    # 6. Solve for the potential in Fourier space
    # The Fourier transform of ∇²U is -k² * U_fft
    # So, -k² * U_fft = 4πG * rho_fft  =>  U_fft = -4πG * rho_fft / k²
    potential_fft = - (4 * np.pi * const.G_CONST * rho_fft) / k_sq
    
    # 7. Restore the DC component to zero (potential is relative)
    potential_fft[0, 0, 0] = 0
    
    # 8. Perform the inverse 3D FFT to get the potential in real space
    potential_real = np.real(fftpack.ifftn(potential_fft))
    
    print(" -> Gravitational potential U_bar calculated.")
    return potential_real


def solve_phi_field_3d_relax(U_bar, grid_step, tol=1e-5, max_iter=1000, omega=1.5):
    """
    Solves the 3D Klein-Gordon-like equation for the Φ field using the
    Successive Over-Relaxation (SOR) method.
    ∇²Φ - m_eff²Φ = 0
    
    Args:
        U_bar (np.array): The 3D gravitational potential of baryons.
        grid_step (float): The distance between grid points.
        tol (float): The convergence tolerance.
        max_iter (int): Maximum number of iterations.
        omega (float): The relaxation parameter (1 < omega < 2 for over-relaxation).
        
    Returns:
        np.array: The 3D scalar field Φ solution.
    """
    print("Solving 3D equation for Φ field using SOR method...")
    
    # 1. Calculate the position-dependent effective mass squared
    m_eff_sq = const.M0_SQ_EFF - 2 * const.BETA_SH_UNIVERSAL * U_bar
    # Ensure mass squared is not negative (physical constraint)
    m_eff_sq[m_eff_sq < 0] = 1e-6 

    # 2. Initialize the Φ field. A good initial guess is important.
    phi = np.zeros_like(U_bar)
    
    # 3. Main SOR iteration loop
    h_sq = grid_step**2
    
    for it in range(max_iter):
        phi_old_max = np.max(phi)
        
        # Loop over all *interior* points of the grid
        for i in range(1, phi.shape[0] - 1):
            for j in range(1, phi.shape[1] - 1):
                for k in range(1, phi.shape[2] - 1):
                    
                    # Standard finite difference stencil for the Laplacian
                    laplacian_term = (phi[i+1,j,k] + phi[i-1,j,k] +
                                      phi[i,j+1,k] + phi[i,j-1,k] +
                                      phi[i,j,k+1] + phi[i,j,k-1])
                    
                    # The update formula derived from the discretized equation
                    term_mass = h_sq * m_eff_sq[i,j,k]
                    
                    # Value from the Jacobi method
                    phi_jacobi = laplacian_term / (6.0 + term_mass)
                    
                    # Apply over-relaxation
                    phi[i,j,k] = (1 - omega) * phi[i,j,k] + omega * phi_jacobi

        # A simple central "source" condition to break symmetry and start the field
        # This simulates the field responding to the deepest part of the potential well
        center_idx = phi.shape[0] // 2
        phi[center_idx, center_idx, center_idx] = 1.0 # Anchor the solution's scale

        # 4. Check for convergence every 20 iterations
        if it % 20 == 0:
            current_max = np.max(phi)
            # Convergence is reached when the change in the solution becomes very small
            error = abs(current_max - phi_old_max)
            print(f"  Iteration: {it}, Max Φ: {current_max:.4f}, Change: {error:.2e}")
            if error < tol and it > 50:
                print(f"Convergence reached at iteration {it}!")
                break
    
    if it == max_iter - 1:
        print("Warning: Maximum number of iterations reached. Solution may not have fully converged.")

    print("Φ field calculation finished.")
    return phi
