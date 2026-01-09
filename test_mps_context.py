import torch
import tensorcircuit as tc

import numpy as np
tc.set_backend("pytorch")
tc.set_dtype("complex64")

# Monkey patch to fix MPS float64 issue
_orig_convert = tc.backend.convert_to_tensor
def _patched_convert(tensor):
    if isinstance(tensor, np.ndarray):
        if tensor.dtype == np.float64:
            tensor = tensor.astype(np.float32)
        elif tensor.dtype == np.complex128:
            tensor = tensor.astype(np.complex64)
    return _orig_convert(tensor)
tc.backend.convert_to_tensor = _patched_convert

def test_context():
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping test")
        return

    print("Testing context manager...")
    with torch.device("mps"):
        # Create a tensor via backend
        t = tc.backend.convert_to_tensor([1.0])
        print(f"Tensor device: {t.device}")
        
        # Check gate
        g = tc.gates.x().tensor
        # Wait, if gates are instantiated lazily, this checks it.
        # If they were already instantiated (e.g. at import), they might be CPU.
        # But 'x' is a function that returns a gate object.
        # The gate object has a .tensor property.
        print(f"Gate device: {g.device}")
        
        if t.device.type == 'mps' and g.device.type == 'mps':
            print("SUCCESS: Context manager works!")
        else:
            print("FAILURE: Device mismatch")

if __name__ == "__main__":
    test_context()
