import os
import cv2
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, List
import logging
from configparser import ConfigParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Image Preprocessor class for computer vision tasks.
    Includes functions for loading, transforming, and augmenting images.
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.means = np.array(self.config['preprocessing']['means']).reshape(1, 1, 3)
        self.stds = np.array(self.config['preprocessing']['stds']).reshape(1, 1, 3)

    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration file.

        Parameters:
            config_path (str): Path to the configuration file.

        Returns:
            Dict: Configuration settings.
        """
        config = ConfigParser()
        config.read(config_path)
        return dict(config.items('preprocessing'))

    def _validate_image(self, image: np.ndarray) -> bool:
        """
        Validate if the input image has valid dimensions and channels.

        Parameters:
            image (np.ndarray): Input image to be validated.

        Returns:
            bool: True if image is valid, False otherwise.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            logger.error("Invalid image shape. Expected NDIM=3 and channels=3.")
            return False
        return True

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image by subtracting mean and dividing by standard deviation.

        Parameters:
            image (np.ndarray): Input image to be normalized.

        Returns:
            np.ndarray: Normalized image.
        """
        image = image.astype(np.float32)
        image = (image - self.means) / self.stds
        return image

    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Load an image from file and apply preprocessing steps.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Preprocessed image.
        """
        try:
            image = cv2.imread(image_path)
            if self._validate_image(image):
                image = self._normalize_image(image)
                return image
            else:
                logger.error("Invalid image. Preprocessing aborted.")
                return None
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation techniques to the input image.
        Current implementation includes random horizontal flip and random crop.

        Parameters:
            image (np.ndarray): Input image to be augmented.

        Returns:
            np.ndarray: Augmented image.
        """
        try:
            if np.random.rand() < self.config.getfloat('augmentation', 'horizontal_flip_prob'):
                image = cv2.flip(image, 1)

            crop_size = self.config.getint('augmentation', 'crop_size')
            if crop_size < image.shape[0] or crop_size < image.shape[1]:
                x = np.random.randint(image.shape[1] - crop_size + 1)
                y = np.random.randint(image.shape[0] - crop_size + 1)
                image = image[y:y+crop_size, x:x+crop_size, :]

            return image
        except Exception as e:
            logger.error(f"Error during image augmentation: {e}")
            return None

class ImageDataset:
    """
    Image Dataset class for loading and preprocessing images from a dataset.
    """
    def __init__(self, image_paths: List[str], preprocessor: ImagePreprocessor):
        self.image_paths = image_paths
        self.preprocessor = preprocessor

    def __getitem__(self, index: int) -> Dict:
        """
        Get image and its corresponding data at the specified index.

        Parameters:
            index (int): Index of the image in the dataset.

        Returns:
            Dict: Image data and its path.
        """
        image_path = self.image_paths[index]
        image_data = self.preprocessor.load_and_preprocess(image_path)
        if image_data is not None:
            return {'image': image_data, 'path': image_path}
        else:
            raise RuntimeError(f"Failed to load and preprocess image: {image_path}")

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.image_paths)

def velocity_threshold(optical_flow: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply velocity thresholding to the optical flow field.
    This function implements the algorithm described in the research paper.

    Parameters:
        optical_flow (np.ndarray): Optical flow field.
        threshold (float): Velocity threshold value.

    Returns:
        np.ndarray: Thresholded optical flow field.
    """
    mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
    mag = mag / (threshold + 1e-5)
    mag[mag > 1] = 1
    mag[mag < 0] = 0
    return cv2.polarToCart(mag, ang)

def flow_to_image(optical_flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow field to a color image for visualization.

    Parameters:
        optical_flow (np.ndarray): Optical flow field.

    Returns:
        np.ndarray: Color image representation of the optical flow.
    """
    flow_image = cv2.cvtColor(optical_flow, cv2.COLOR_BGR2GRAY)
    flow_image = cv2.applyColorMap(flow_image, cv2.COLORMAP_JET)
    return flow_image

def main():
    # Load configuration
    config_path = 'preprocessing_config.ini'
    preprocessor = ImagePreprocessor(config_path)

    # Example usage
    image_path = 'example.jpg'
    preprocessed_image = preprocessor.load_and_preprocess(image_path)
    if preprocessed_image is not None:
        logger.info("Image loaded and preprocessed successfully.")
        # Augment the image
        augmented_image = preprocessor.augment_image(preprocessed_image)
        if augmented_image is not None:
            logger.info("Image augmented successfully.")

    # Create a dataset
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Replace with actual image paths
    dataset = ImageDataset(image_paths, preprocessor)

    # Iterate over the dataset
    for item in dataset:
        image = item['image']
        path = item['path']
        logger.info(f"Processed image: {path}")

    # Example usage of velocity_threshold function
    optical_flow = np.random.rand(240, 320, 2) * 20  # Example optical flow field
    threshold = 5  # Example threshold value
    thresholded_flow = velocity_threshold(optical_flow, threshold)
    logger.info("Velocity thresholding applied successfully.")

    # Convert optical flow to image for visualization
    flow_image = flow_to_image(thresholded_flow)
    cv2.imwrite('optical_flow.jpg', flow_image)
    logger.info("Optical flow saved as image successfully.")

if __name__ == '__main__':
    main()