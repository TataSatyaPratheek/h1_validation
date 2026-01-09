import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ParityDataset(Dataset):
    """
    Synthetic dataset for Phase 3: The Test Flight.
    Input: N-bit strings (represented as floats for CNN compat if needed, or simple vectors).
    Label: Parity of the bits (0 if even, 1 if odd).
    """
    def __init__(self, n_bits=4, num_samples=1000):
        self.n_bits = n_bits
        self.num_samples = num_samples
        
        # Generate random bit strings
        self.data = torch.randint(0, 2, (num_samples, n_bits)).float()
        
        # Compute parity
        self.labels = torch.sum(self.data, dim=1).long() % 2
        
        # Reshape to "image-like" 1xNxN if needed for current CNN, 
        # or we can modify the training loop.
        # Let's reshape to 1x2x2 for a 4-bit parity task.
        self.data_reshaped = self.data.view(num_samples, 1, int(np.sqrt(n_bits)), int(np.sqrt(n_bits)))
        # Actually, let's just make them 3x64x64 dummy images to reuse train.py exactly.
        # We will embed the bits in the first 4 pixels.
        self.full_images = torch.zeros((num_samples, 3, 64, 64))
        for i in range(num_samples):
            # Embed bits into R channel top-left 2x2
            bits = self.data[i]
            self.full_images[i, 0, 0, 0] = bits[0]
            self.full_images[i, 0, 0, 1] = bits[1]
            self.full_images[i, 0, 1, 0] = bits[2]
            self.full_images[i, 0, 1, 1] = bits[3]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.full_images[idx], self.labels[idx]

def get_parity_dataloaders(batch_size=32):
    train_dataset = ParityDataset(num_samples=2000)
    test_dataset = ParityDataset(num_samples=500)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    tl, _ = get_parity_dataloaders()
    imgs, lbls = next(iter(tl))
    print(f"Image Batch Shape: {imgs.shape}")
    print(f"Sample Bits (Top Left): {imgs[0, 0, :2, :2]}")
    print(f"Sample Label: {lbls[0]}")
