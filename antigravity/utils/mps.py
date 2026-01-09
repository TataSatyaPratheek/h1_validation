import torch
import tensorcircuit as tc
import numpy as np

def setup_mps_environment():
    """
    Configures the environment for Apple Silicon (MPS).
    1. Sets default torch device to MPS.
    2. Patches TensorCircuit to handle float64/complex128 incompatibility.
    """
    if torch.backends.mps.is_available():
        torch.set_default_device("mps")
        print("[System] MPS Acceleration Enabled.")
        
        # Monkey patch to fix MPS float64 issue
        # TensorCircuit defaults to float64/complex128 which MPS does not support
        _orig_convert = tc.backend.convert_to_tensor
        def _patched_convert(tensor):
            if isinstance(tensor, np.ndarray):
                if tensor.dtype == np.float64:
                    tensor = tensor.astype(np.float32)
                elif tensor.dtype == np.complex128:
                    tensor = tensor.astype(np.complex64)
            return _orig_convert(tensor)
        tc.backend.convert_to_tensor = _patched_convert
        print("[System] TensorCircuit float64->float32 Patch Applied.")
    else:
        print("[System] MPS not available. Using CPU.")

def get_generator(device_str):
    """
    Returns a torch.Generator for the specified device.
    Essential for DataLoader on MPS.
    """
    if device_str == 'mps':
        return torch.Generator(device='mps')
    return torch.Generator(device='cpu')
