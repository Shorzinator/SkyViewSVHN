import os

import h5py
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


def extract_labels(bbox, max_length=5):
    """
    Extract labels from bounding box data and pad sequences to a fixed length.
    Args:
        bbox: List of bounding box dictionaries.
        max_length: Maximum length of the sequence.
    Returns:
        Numpy array of labels.
    """
    labels = []
    for box in bbox:
        label = [int(h[0]) if h[0] != 10 else 0 for h in box['label']]  # Convert 10 -> 0 for digit '0'
        labels.append(label)
    return pad_sequences(labels, maxlen=max_length, padding='post')


def prepare_data_for_model(images, bboxes, test_size=0.2, random_state=42):
    """
    Prepares the dataset for training and testing.

    Args:
        images: List of PIL image objects.
        bboxes: List of bounding box information.
        test_size: Fraction of data to be used as test set.
        random_state: Random state for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Training and testing splits.
    """
    # Preprocess images: resize, normalize, etc.
    processed_images = preprocess_images(images)  # Assuming this function returns np.array

    # Extract and preprocess labels
    labels = extract_labels(bboxes)

    # Split data into training and testing
    return train_test_split(processed_images, labels, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    data_dir = get_path_from_root("data", "trial")  # Modify as needed
    images, bboxes = load_svhn_data(data_dir)

    # Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data_for_model(images, bboxes)
