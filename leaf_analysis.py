# -*- coding: utf-8 -*-
import cv2
import numpy as np

def load_and_process_image(image_path):
    """
    Loads the image, detects the leaf and infected areas.
    
    Args:
        image_path (str): Path to the image (can contain UTF-8 characters).
    
    Returns:
        tuple: (image, leaf_mask, infected_mask)
        
    Raises:
        Exception: If the image cannot be loaded or processed.
    """
    try:
        # Load the image handling UTF-8 paths
        with open(image_path, 'rb') as f:
            image_data = np.frombuffer(f.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Unable to load image from: {image_path}")

        # Convert the image from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to HSV for segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        # Detect the leaf (exclude white background)
        leaf_mask = saturation > 25  # Threshold for saturation to exclude white background
        leaf_mask = leaf_mask.astype(np.uint8)

        # Detect infected areas (yellow-brown)
        lower_hue = np.array([15, 50, 50])  # Range for yellow-brown
        upper_hue = np.array([30, 255, 255])
        infected_mask = cv2.inRange(hsv, lower_hue, upper_hue)
        infected_mask = infected_mask & leaf_mask  # Apply the leaf mask

        return image, leaf_mask, infected_mask

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def calculate_severity(leaf_mask, infected_mask):
    """
    Calculates the percentage of the infected area.
    
    Args:
        leaf_mask (ndarray): Mask of the leaf.
        infected_mask (ndarray): Mask of the infected areas.
    
    Returns:
        float: Percentage of severity.
        
    Raises:
        ValueError: If the leaf mask is empty.
    """
    total_leaf_pixels = np.count_nonzero(leaf_mask)
    if total_leaf_pixels == 0:
        raise ValueError("The leaf mask is empty, unable to calculate severity.")
    
    infected_pixels = np.count_nonzero(infected_mask)
    severity_percentage = (infected_pixels / total_leaf_pixels) * 100
    return severity_percentage