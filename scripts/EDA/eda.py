import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from tqdm import tqdm

from scripts.utility.path_utils import get_path_from_root

# Paths to the data directories
train_data_path = get_path_from_root('data\\train')
test_data_path = get_path_from_root('data\\test')
extra_data_path = get_path_from_root('data\\extra')
trial_data_path = get_path_from_root('data\\trial')


# Define the function to load bounding box data
def get_bounding_boxes(data_path, subset_size):
    # Open the file
    with h5py.File(os.path.join(data_path, 'digitStruct.mat'), 'r') as digit_struct_mat:
        # Prepare arrays to store bounding box information
        all_boxes = []

        for i in tqdm(range(min(subset_size, len(digit_struct_mat['/digitStruct/name']))),
                      desc='Loading bounding boxes'):
            bbox = {}
            # Get name and labels
            name_ref = digit_struct_mat['/digitStruct/name'][i][0]
            bbox['name'] = ''.join([chr(c[0]) for c in digit_struct_mat[name_ref][:]])

            bbox_item_ref = digit_struct_mat['/digitStruct/bbox'][i][0]
            bbox_item = digit_struct_mat[bbox_item_ref]

            for attr in ['label', 'left', 'top', 'height', 'width']:
                if isinstance(bbox_item[attr], h5py.Reference):
                    # For individual references, we should first dereference.
                    data_ref = digit_struct_mat[bbox_item[attr]]
                    if isinstance(data_ref, h5py.Dataset):
                        values = data_ref[:].astype('float32')
                    else:
                        # If it's a reference, dereference to get the actual data.
                        ref_dereferenced = digit_struct_mat[data_ref][()]
                        values = np.array(ref_dereferenced).astype('float32')
                else:
                    # For datasets containing multiple references, we loop over them.
                    values = []
                    for ref in bbox_item[attr]:
                        dereferenced_values = digit_struct_mat[ref][:].astype('float32')
                        values.append(dereferenced_values)

                    # Convert a list of arrays to a single flat array.
                    values = np.concatenate(values)

                # Assign to the bbox dictionary.
                bbox[attr] = values.flatten() if values.ndim > 1 else values

            all_boxes.append(bbox)

        return all_boxes


# Function to plot example images with bounding boxes
def plot_example_images(data_path, bounding_boxes, num_examples=5):
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    for ax, bbox in zip(axes, bounding_boxes[:num_examples]):
        # Assuming image names are numbers starting from 1 in the dataset
        img_path = os.path.join(data_path, f'{bbox["name"]}')
        image = Image.open(img_path)
        ax.imshow(image)
        ax.axis('off')
        # Iterate over bounding boxes and draw them
        for i in range(len(bbox['label'])):
            # Draw the rectangle
            rect = Rectangle((bbox['left'][i], bbox['top'][i]), bbox['width'][i], bbox['height'][i], edgecolor='red',
                             facecolor='none')
            ax.add_patch(rect)
    plt.show()


# Function to count the distribution of number lengths in images
def count_number_lengths(bounding_boxes):
    lengths = [len(bbox['label']) for bbox in bounding_boxes]
    return lengths


# Function to plot the distribution of number lengths
def plot_length_distribution(lengths):
    plt.hist(lengths, bins=np.arange(0.5, max(lengths) + 1.5, 1), rwidth=0.9)
    plt.title('Distribution of Number Lengths in Images')
    plt.xlabel('Length of Number')
    plt.ylabel('Frequency')
    plt.xticks(range(1, max(lengths) + 1))
    plt.show()


if __name__ == "__main__":
    print('Analyzing test data...')
    train_bounding_boxes = get_bounding_boxes(test_data_path, subset_size=100)
    train_lengths = count_number_lengths(train_bounding_boxes)
    plot_length_distribution(train_lengths)
    plot_example_images(train_data_path, train_bounding_boxes)
