import numpy as np
from PIL import Image


def preprocess_images(images, target_size=(32, 32)):
    """
    Preprocess images for model input

    Args:
        images: List of PIL images.
        target_size: Target size to resize images.

    Returns:
        preprocessed_images: Numpy array of preprocessed images.

    """
    preprocessed_images = []

    for img in images:
        # Resize images
        img = img.resize(target_size, Image.ANTIALIAS)

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255, 0
        preprocessed_images.append(img_array)

    return np.array(preprocessed_images)
