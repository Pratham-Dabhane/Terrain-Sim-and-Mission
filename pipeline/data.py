"""
DES Dataset Loader for Terrain Generation
Loads and preprocesses Digital Elevation Model (DEM) heightmaps for training.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional, List
import glob
import logging

logger = logging.getLogger(__name__)

class DESDataset(Dataset):
    """
    Dataset loader for Digital Elevation Model (DEM) heightmaps.
    
    Args:
        data_dir: Directory containing heightmap images (PNG, TIFF, etc.)
        image_size: Target size for heightmaps (default: 256)
        normalize: Whether to normalize heightmaps to [0, 1] range
        augment: Whether to apply data augmentation
        file_extensions: List of valid file extensions
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        normalize: bool = True,
        augment: bool = True,
        file_extensions: List[str] = ['.png', '.tiff', '.tif', '.jpg', '.jpeg']
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # Find all heightmap files
        self.heightmap_paths = []
        for ext in file_extensions:
            pattern = os.path.join(data_dir, f"**/*{ext}")
            self.heightmap_paths.extend(glob.glob(pattern, recursive=True))
        
        if len(self.heightmap_paths) == 0:
            raise ValueError(f"No heightmap files found in {data_dir}")
        
        logger.info(f"Found {len(self.heightmap_paths)} heightmaps in {data_dir}")
        
        # Setup transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transforms for preprocessing"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        ]
        
        # Data augmentation (only during training)
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90, interpolation=transforms.InterpolationMode.BILINEAR),
            ])
        
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if self.normalize:
            # Normalize to [0, 1] range
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.heightmap_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a heightmap tensor.
        
        Returns:
            torch.Tensor: Normalized heightmap of shape (1, H, W)
        """
        heightmap_path = self.heightmap_paths[idx]
        
        try:
            # Load image
            image = Image.open(heightmap_path).convert('L')  # Convert to grayscale
            
            # Apply transforms
            heightmap = self.transform(image)
            
            return heightmap
            
        except Exception as e:
            logger.warning(f"Error loading {heightmap_path}: {e}")
            # Return a random heightmap as fallback
            return torch.randn(1, self.image_size, self.image_size)
    
    def get_sample_batch(self, batch_size: int = 4) -> torch.Tensor:
        """Get a sample batch for testing"""
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        batch = torch.stack([self[i] for i in indices])
        return batch


class DESDataModule:
    """
    Data module for easy train/val split and dataloader creation.
    Memory-efficient for low VRAM systems.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,  # Small batch for GTX 1650
        image_size: int = 256,
        val_split: float = 0.1,
        num_workers: int = 0,  # 0 for Windows compatibility
        pin_memory: bool = False,  # Disabled for low RAM
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self):
        """Setup train and validation datasets"""
        # Full dataset
        full_dataset = DESDataset(
            data_dir=self.data_dir,
            image_size=self.image_size,
            normalize=True,
            augment=True
        )
        
        # Train/val split
        total_size = len(full_dataset)
        val_size = int(self.val_split * total_size)
        train_size = total_size - val_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Disable augmentation for validation
        val_dataset_base = DESDataset(
            data_dir=self.data_dir,
            image_size=self.image_size,
            normalize=True,
            augment=False
        )
        
        # Replace val dataset with non-augmented version
        self.val_dataset.dataset = val_dataset_base
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Ensure consistent batch sizes
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


def create_synthetic_data(output_dir: str, num_samples: int = 100):
    """
    Create synthetic heightmap data for testing when real DES data is not available.
    
    Args:
        output_dir: Directory to save synthetic heightmaps
        num_samples: Number of synthetic samples to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating {num_samples} synthetic heightmaps in {output_dir}")
    
    for i in range(num_samples):
        # Generate random heightmap using Perlin-like noise
        size = 256
        x = np.linspace(0, 4, size)
        y = np.linspace(0, 4, size)
        X, Y = np.meshgrid(x, y)
        
        # Multi-octave noise
        heightmap = np.zeros((size, size))
        for octave in range(4):
            freq = 2 ** octave
            amp = 1.0 / (2 ** octave)
            heightmap += amp * np.sin(freq * X) * np.cos(freq * Y)
        
        # Add some random mountains/valleys
        num_features = np.random.randint(1, 4)
        for _ in range(num_features):
            cx, cy = np.random.randint(0, size, 2)
            radius = np.random.randint(20, 80)
            height = np.random.uniform(-0.5, 1.0)
            
            # Create circular feature
            for x in range(max(0, cx - radius), min(size, cx + radius)):
                for y in range(max(0, cy - radius), min(size, cy + radius)):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius:
                        factor = np.cos(np.pi * dist / (2 * radius))
                        heightmap[y, x] += height * factor
        
        # Normalize to [0, 255]
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        heightmap = (heightmap * 255).astype(np.uint8)
        
        # Save as PNG
        img = Image.fromarray(heightmap, mode='L')
        img.save(os.path.join(output_dir, f"heightmap_{i:04d}.png"))
    
    logger.info(f"Synthetic data created successfully")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data if needed
    synthetic_dir = "data/synthetic_heightmaps"
    if not os.path.exists(synthetic_dir):
        create_synthetic_data(synthetic_dir, num_samples=50)
    
    # Test dataset
    try:
        dataset = DESDataset(synthetic_dir, image_size=256)
        print(f"Dataset size: {len(dataset)}")
        
        # Test sample
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        # Test datamodule
        datamodule = DESDataModule(synthetic_dir, batch_size=2)
        datamodule.setup()
        
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have heightmap data in the specified directory")