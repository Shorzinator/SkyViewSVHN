import os

import numpy as np
import h5py
from PIL import Image

from scripts.preprocessing.preprocessing import preprocess_images
from scripts.utility.path_utils import get_path_from_root


def load_digit_struct(file_path):
    """
    Load the digitStruct.mat file which contains bounding ox information.

    Args:
        file_path: Path to the digitStruct.mat file

    Returns:
        digit_struct: Struct with bounding box information

    """
    with h5py.File(file_path, "r") as file:
        digit_struct = file["digitStruct"]["bbox"]
    return digit_struct


def bbox_helper(attr, file):
    """
    Helper function to handle the attribute extraction from the bounding box structure.
    Args:
        attr: The attribute to extract.
        file: The HDF5 file reference.

    Returns:
        Attribute Value.
    """

    if len(attr) > 1:
        attr = [file[attr.value[j].item()].value[0][0] for j in range(len(attr))]
    else:
        attr = [attr.value[0][0]]

    return attr


def get_bounding_boxes(digit_struct, file, index):
    """
    Extract and return the bounding box information for a given image index.
    Args:
        digit_struct: The digit structure reference.
        file: The HDF5 file reference.
        index: Index of the Image.

    Returns:
        Bounding box information for the image.

    """
    bbox = {}
    bb = digit_struct[index].item()

    bbox["height"] = bbox_helper(file[bb]["height"], file)
    bbox["left"] = bbox_helper(file[bb]["left"], file)
    bbox["top"] = bbox_helper(file[bb]["top"], file)
    bbox["width"] = bbox_helper(file[bb]["width"], file)

    return bbox


def load_svhn_data(data_dir, subset="extra"):
    """
    Load SVHN dataset images and bounding box information

    Args:
        data_dir: Path to the directory containing SVHN files
        subset: Subset of the SVHN dataset to use

    Returns:
        images: List of images
        bbox: List of bounding box information for each image.
    """

    images = []
    bboxes = []

    # Load the digitStruct.mat file
    digit_struct = load_digit_struct(os.path.join(data_dir, subset, "digitStruct.mat"))

    with h5py.File(os.path.join(data_dir, subset, "digitStruct.mat"), "r") as file:
        for i in range(len(digit_struct)):
            # Load the image
            img_name = "".join(chr(c[0]) for c in file[file["digitStruct"]["name"][i][0].value])
            img = Image.open(os.path.join(data_dir, subset, img_name))
            images.append(img)

            # Load bounding box
            bbox = get_bounding_boxes(digit_struct, file, i)
            bboxes.append(bbox)

    return images, bboxes


if __name__ == "__main__":
    data_dir = get_path_from_root("data", "trial")  # Modify as needed
    images, bboxes = load_svhn_data(data_dir)
    images = preprocess_images(images)
