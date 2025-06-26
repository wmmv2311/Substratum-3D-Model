# === Physical Constants ===
# Gravitational constant G in units of (kpc/M_sun) * (km/s)^2
# These units are convenient for galactic dynamics.
G_CONST = 4.302e-6

# === Model Parameters ===
# The universal constant for the substratum response
BETA_SH_UNIVERSAL = 8.59e-4

# The "true" mass squared of the phi-quantum in cosmological units.
# This is a free parameter and needs to be constrained by observations.
# We will start with a value that works well on galactic scales.
# Units: 1/kpc^2
M0_SQ_EFF = 0.008

# === Numerical Grid Parameters ===
# This parameter can be changed for higher/lower resolution runs.
# Possible values: 'test' (for a quick test) or 'full_run'.
RESOLUTION = 'test'

if RESOLUTION == 'full_run':
    # Number of points along each axis (e.g., from -X to +X)
    N_POINTS = 400  
else: # test resolution
    N_POINTS = 100
