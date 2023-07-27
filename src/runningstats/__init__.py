# Check if CuPy installed or not
CUPY_INSTALLED = False
try:
    import cupy
    CUPY_INSTALLED = True
except:
    pass

# Imports
from .statstracker import StatsTracker