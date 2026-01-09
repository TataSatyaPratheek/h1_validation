import torch
import torchvision
import torchvision.transforms as transforms
import os

from torch.utils.data import Dataset, DataLoader

class ParityDataset(Dataset):
    def __init__(self, n_bits=4, num_samples=1000):
        self.n_bits = n_bits
        self.num_samples = num_samples
        bits = torch.randint(0, 2, (num_samples, n_bits)).float()
        self.labels = torch.sum(bits, dim=1).long() % 2
        self.full_images = torch.zeros((num_samples, 3, 64, 64))
        for i in range(num_samples):
            self.full_images[i, 0, 0, 0] = bits[i, 0]
            self.full_images[i, 0, 0, 1] = bits[i, 1]
            self.full_images[i, 0, 1, 0] = bits[i, 2]
            self.full_images[i, 0, 1, 1] = bits[i, 3]

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.full_images[idx], self.labels[idx]

def get_dataloaders(data_dir='./data', batch_size=32, image_size=64, num_workers=0, dataset_name='cifar10', generator_device='cpu', subset_size=None):
    
    if dataset_name == "parity":
        gen = torch.Generator(device=generator_device) if generator_device == 'mps' else None
        train_loader = DataLoader(ParityDataset(num_samples=2000), batch_size=batch_size, shuffle=True, generator=gen)
        val_loader = DataLoader(ParityDataset(num_samples=500), batch_size=batch_size, shuffle=False, generator=gen)
        return train_loader, val_loader

    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    transform = transforms.Compose(transform_list)

    if dataset_name.lower() == 'imagenet':
        # Expects standard ImageNet structure: train/ and val/ subfolders
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        
        if not os.path.exists(traindir):
            print(f"ImageNet training dir not found at {traindir}. Falling back to CIFAR10.")
            dataset_name = 'cifar10'
        else:
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transform=transform
            )
            val_dataset = torchvision.datasets.ImageFolder(
                valdir,
                transform=transform
            )
    
    # Check for CIFAR10 fallback or explicit request
    if dataset_name.lower() == 'cifar10':
        print(f"Loading CIFAR10 from {data_dir}...")
        # Torchvision checks checksums if valid
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name.lower() == 'fake':
        print("Loading FakeData for rapid verification...")
        train_dataset = torchvision.datasets.FakeData(
            size=100 if subset_size is None else subset_size, 
            image_size=(3, image_size, image_size), num_classes=10, transform=transforms.ToTensor())
        val_dataset = torchvision.datasets.FakeData(
            size=20 if subset_size is None else max(20, subset_size//5), 
            image_size=(3, image_size, image_size), num_classes=10, transform=transforms.ToTensor())

    # Subset Logic
    if subset_size is not None and dataset_name.lower() != 'fake':
        print(f"Subsetting dataset to {subset_size} samples...")
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        # Keep validation small too relative to subset
        val_indices = torch.randperm(len(val_dataset))[:max(100, subset_size//5)]
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    g = torch.Generator(device=generator_device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=g)
        
    return train_loader, val_loader

if __name__ == "__main__":
    train, val = get_dataloaders(dataset_name='cifar10', batch_size=4)
    x, y = next(iter(train))
    print(f"Batch shape: {x.shape}, Label shape: {y.shape}")
