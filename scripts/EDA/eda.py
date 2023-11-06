import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image

# Paths to the data directories
train_data_path = 'SkyViewSVHN\\data\\train'
test_data_path = 'SkyViewSVHN\\data\\test'
extra_data_path = 'SkyViewSVHN\\data\\extra'


# Function to load the digitStruct.mat file
def load_mat_file(data_path):
    # Load the .mat file
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'digitStruct.mat'))
    return mat_data['digitStruct']


# Function to count the distribution of number lengths in images
def count_number_lengths(digit_struct_bboxes):
    lengths = []
    for i in range(len(digit_struct_bboxes)):
        lengths.append(len(digit_struct_bboxes[i]['label'][0]))

    return lengths


# Function to plot the distribution of number lengths
def plot_length_distribution(lengths):
    plt.hist(lengths, bin=np.arange(1, max(lengths) + 1) - 0.5, rwidth=0.9)
    plt.title('Distribution of number Lengths')
    plt.xlabel('Length of NUmber')
    plt.ylabel('Frequency')
    plt.xticks(range(1, max(lengths) + 1))
    plt.show()


# Function to plot images
def plot_images(data_path, digit_struct_bboxes, num_examples=5):
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    for i in range(num_examples):
        bbox = digit_struct_bboxes[i]
        img_name = bbox['name']
        img_path = os.path.join(data_path, img_name)
        image = Image.open(img_path)


if __name__ == "__main__":
    # Load digitStruct for train, test, and extra datasets
    train_digit_struct = load_mat_file(train_data_path)
    test_digit_struct = load_mat_file(test_data_path)
    extra_digit_struct = load_mat_file(extra_data_path)

    # Count and plot the number length distribution for train dataset
    train_lengths = count_number_lengths(train_digit_struct)
    plot_length_distribution(train_lengths)

    # Plot images from the train dataset
    plot_images(train_data_path, train_digit_struct)
