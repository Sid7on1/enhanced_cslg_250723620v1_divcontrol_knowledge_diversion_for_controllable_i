import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import json
import yaml
import argparse
from typing import List, Tuple, Dict
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants and configuration
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'data_dir': './data',
    'batch_size': 32,
    'image_size': 256,
    'num_workers': 4,
    'shuffle': True
}

class ImageDataset(Dataset):
    """
    Custom dataset class for loading and batching images.
    """
    def __init__(self, data_dir: str, image_size: int, transform=None):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Directory containing the image data.
            image_size (int): Size to resize the images to.
            transform (callable, optional): Optional transform to apply to the images.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.image_paths = []
        self.label_paths = []

        # Load image and label paths from the data directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    self.image_paths.append(image_path)
                    label_path = os.path.join(root, file.replace('.jpg', '.json').replace('.png', '.json'))
                    self.label_paths.append(label_path)

    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """
        Return the image and label at the specified index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            Tuple: Image and label at the specified index.
        """
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        # Load the image
        image = Image.open(image_path)
        image = image.resize((self.image_size, self.image_size))

        # Load the label
        with open(label_path, 'r') as f:
            label = json.load(f)

        # Apply the transform to the image
        if self.transform:
            image = self.transform(image)

        # Convert the label to a tensor
        label = torch.tensor(label)

        return image, label

class DataLoader:
    """
    Custom data loader class for loading and batching images.
    """
    def __init__(self, config: Dict):
        """
        Initialize the data loader.

        Args:
            config (Dict): Configuration dictionary.
        """
        self.config = config
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.image_size = config['image_size']
        self.num_workers = config['num_workers']
        self.shuffle = config['shuffle']

        # Create the dataset
        self.dataset = ImageDataset(self.data_dir, self.image_size)

        # Create the data loader
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def load_data(self):
        """
        Load the data and return the data loader.
        """
        return self.data_loader

def load_config(config_file: str = CONFIG_FILE) -> Dict:
    """
    Load the configuration from the specified file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        Dict: Configuration dictionary.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Merge the default configuration with the loaded configuration
    config = {**DEFAULT_CONFIG, **config}

    return config

def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Load and batch image data')
    parser.add_argument('--config', type=str, default=CONFIG_FILE, help='Path to the configuration file')
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Create the data loader
    data_loader = DataLoader(config)

    # Load the data
    data_loader.load_data()

if __name__ == '__main__':
    main()