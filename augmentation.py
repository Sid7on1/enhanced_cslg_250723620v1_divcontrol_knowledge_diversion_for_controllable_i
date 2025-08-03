import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import random
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAugmentation:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def random_crop(self, image, crop_size):
        image = Image.fromarray(image)
        width, height = image.size
        x = random.randint(0, width - crop_size)
        y = random.randint(0, height - crop_size)
        return image.crop((x, y, x + crop_size, y + crop_size))

    def random_flip(self, image):
        return np.fliplr(image)

    def random_rotate(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def random_affine(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def apply_transform(self, image):
        image = self.transform(image)
        return image

    def apply_random_transform(self, image):
        image = self.random_crop(image, self.config['image_size'])
        image = self.random_flip(image)
        image = self.random_rotate(image, random.randint(-30, 30))
        image = self.random_affine(image, random.randint(-30, 30))
        image = self.apply_transform(image)
        return image

class DataAugmentationDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image

class DataAugmentationConfig:
    def __init__(self, image_size=256, batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size

class DataAugmentationService:
    def __init__(self, config):
        self.config = config
        self.data_augmentation = DataAugmentation(config)
        self.dataset = DataAugmentationDataset(config.data, transform=self.data_augmentation.apply_random_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

    def get_dataloader(self):
        return self.dataloader

    def get_data_augmentation(self):
        return self.data_augmentation

# Example usage
if __name__ == "__main__":
    config = DataAugmentationConfig(image_size=256, batch_size=32)
    data = [np.random.rand(256, 256, 3) for _ in range(100)]
    service = DataAugmentationService(config)
    dataloader = service.get_dataloader()
    data_augmentation = service.get_data_augmentation()

    for batch in dataloader:
        images = batch
        for i, image in enumerate(images):
            logger.info(f"Image {i} shape: {image.shape}")
            logger.info(f"Image {i} mean: {image.mean()}")
            logger.info(f"Image {i} std: {image.std()}")