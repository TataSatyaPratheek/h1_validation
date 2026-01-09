import torch
from torch.utils.data import DataLoader, TensorDataset

def test_dataloader():
    if not torch.backends.mps.is_available():
        return

    print("Setting default device to MPS")
    torch.set_default_device("mps")
    
    # Create MPS generator
    g = torch.Generator(device='mps')
    g.manual_seed(42)
    
    # Dataset
    data = torch.randn(10, 2).to("cpu") # Keep data on CPU initially
    ds = TensorDataset(data)
    
    # Loader with generator
    # Setting worker_init_fn might be needed? 
    # Or just passing generator to DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=True, generator=g, num_workers=0)
    
    print("Iterating loader...")
    try:
        for batch in loader:
            print(f"Batch device: {batch[0].device}")
        print("SUCCESS")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_dataloader()
