import numpy as np
from pathlib import Path

def save_data(data, path):
    """Save numpy array to file."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    np.save(path, data)

def load_data(path):
    """Load numpy array from file."""
    return np.load(path)