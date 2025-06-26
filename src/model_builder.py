import numpy as np
# The '.' before 'constants' means we are importing from the same package (the 'src' folder)
from . import constants as const 

def create_3d_grid(galaxy_params):
    """
    Creates a 3D Cartesian grid based on galaxy size.
    
    Args:
        galaxy_params (dict): A dictionary containing galaxy parameters,
                              including 'grid_max_dim'.
                              
    Returns:
        dict: A dictionary containing the grid coordinates (x, y, z axes and
              X, Y, Z meshgrids) and the grid step size.
    """
    max_dim = galaxy_params['grid_max_dim']
    n_points = const.N_POINTS
    
    x = np.linspace(-max_dim, max_dim, n_points)
    y = np.linspace(-max_dim, max_dim, n_points)
    z = np.linspace(-max_dim, max_dim, n_points)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    grid_info = {
        'x': x, 'y': y, 'z': z,
        'X': X, 'Y': Y, 'Z': Z,
        'step': x[1] - x[0]
    }
    return grid_info

def get_galaxy_params(name='NGC1097'):
    """
    Returns a dictionary with physical parameters for a given galaxy.
    
    Args:
        name (str): The name of the galaxy.
        
    Returns:
        dict: A dictionary containing the physical parameters.
    """
    if name == 'NGC1097':
        # Parameters are illustrative, based on typical values for this galaxy type
        return {
            'grid_max_dim': 30, # kpc, the size of our simulation box
            # --- Baryonic Components (Masses in M_sun) ---
            'M_disk': 6.0e10,
            'M_bar': 3.5e10,
            'M_bulge': 1.0e10,
            'M_gas': 0.5e10,
            # --- Scale lengths (in kpc) ---
            'R_d_disk': 4.5,    # Disk scale length
            'z_d_disk': 0.2 * 4.5, # Disk scale height
            # For the bar (approximated as a flattened ellipsoid/triaxial system)
            'a_bar': 5.0, # Major axis (along x-axis)
            'b_bar': 1.5, # Intermediate axis (along y-axis)
            'c_bar': 0.4, # Minor axis (vertical, along z-axis)
            # For the bulge
            'a_bulge': 0.7 
        }
    else:
        # We can add more galaxies here in the future
        raise ValueError(f"Galaxy '{name}' is not yet implemented in the database.")

def hernquist_profile(X, Y, Z, M, a):
    """Calculates 3D density for a spherical Hernquist bulge."""
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r[r == 0] = 1e-6 # Avoid division by zero at the very center
    rho = (M / (2 * np.pi)) * (a / (r * (r + a)**3))
    return rho

def exponential_disk_profile(X, Y, Z, M, R_d, z_d):
    """Calculates 3D density for an exponential disk."""
    R = np.sqrt(X**2 + Y**2)
    # Normalization constant for the density
    rho0 = M / (4 * np.pi * R_d**2 * z_d)
    rho = rho0 * np.exp(-R / R_d) * (1 / np.cosh(Z / z_d))**2
    return rho

def ferrers_bar_profile(X, Y, Z, M, a, b, c):
    """
    Calculates 3D density for a triaxial Ferrers bar (n=2).
    The bar is assumed to be aligned with the x-axis.
    """
    m_sq = (X/a)**2 + (Y/b)**2 + (Z/c)**2
    rho = np.zeros_like(X)
    
    # Density is non-zero only inside the ellipsoid defined by m_sq < 1
    inside_indices = m_sq < 1
    
    # Central density normalization for n=2 Ferrers bar
    rho0 = (105 / (32 * np.pi)) * (M / (a*b*c))
    
    rho[inside_indices] = rho0 * (1 - m_sq[inside_indices])**2
    return rho

def build_baryonic_density(galaxy_name='NGC1097'):
    """
    Builds the total 3D baryonic density grid for a specified galaxy.
    
    Args:
        galaxy_name (str): The name of the galaxy to model.
        
    Returns:
        tuple: A tuple containing the total 3D density grid (numpy array)
               and the grid information dictionary.
    """
    params = get_galaxy_params(galaxy_name)
    grid = create_3d_grid(params)
    
    X, Y, Z = grid['X'], grid['Y'], grid['Z']
    
    print(f"Building baryonic components for {galaxy_name}...")
    
    # Note: We combine the gas mass with the disk mass for simplicity
    print(" -> Building disk component...")
    rho_disk = exponential_disk_profile(X, Y, Z, params['M_disk'] + params['M_gas'], params['R_d_disk'], params['z_d_disk'])
    
    print(" -> Building bar component...")
    rho_bar = ferrers_bar_profile(X, Y, Z, params['M_bar'], params['a_bar'], params['b_bar'], params['c_bar'])
    
    print(" -> Building bulge component...")
    rho_bulge = hernquist_profile(X, Y, Z, params['M_bulge'], params['a_bulge'])
    
    total_rho = rho_disk + rho_bar + rho_bulge
    print("Baryonic density grid successfully built.")
    
    return total_rho, grid
