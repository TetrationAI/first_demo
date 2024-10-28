import os
import os
import numpy as np
from random import shuffle
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split


#Image file loader: 
def read_images(folder_path):
    files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', 'jpeg')):
            files.append(os.path.join(folder_path, file_name))
    return files

def add_name(directory_path, string):
    """
    Add a string to the filename before the file extension.

    Parameters
    ----------
    directory_path : str
        Path to the directory.
    string : str
        String to be added to the filename.
    """
    directory = directory_path

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and string not in filename:
            name, extension = os.path.splitext(filename)
            new_filename = f"{name}{string}{extension}"
            os.rename(file_path, os.path.join(directory, new_filename))


################################## CV MODIFIED GENERATOR FUNCTIONS FROM RAYS-PROJECT ###################################

def get_data_separate(data_path_class1, data_path_class2, img_h, img_w, test_size=0.2):
    """
    Loads images from separate directories for each class and divides them into training and testing sets.

    Parameters
    ----------
    data_path_class1 : Path to the directory for class 1.
    data_path_class2 : Path to the directory for class 2.
    img_h : Image height.
    img_w : Image width.
    test_size : Proportion of the dataset to include in the test split.

    Returns
    -------
    train_data : Nested List
        A nested list containing the loaded images along with their corresponding labels for the training set.
    test_data : Nested List
        A nested list containing the loaded images along with their corresponding labels for the testing set.
    """
    data_class1 = []
    data_class2 = []

    for class_path, label in [(data_path_class1, 0), (data_path_class2, 1)]:
        data_list = os.listdir(class_path)
        images = []
        labels = []
        for item in data_list:
            img = imread(os.path.join(class_path, item), as_gray=True)
            img = resize(img, (img_h, img_w), anti_aliasing=True).astype('float32')
            images.append(np.array(img))
            labels.append(np.array([label]))

        # Combining data for both classes
        if label == 0:
            data_class1.extend(list(zip(images, labels)))
        else:
            data_class2.extend(list(zip(images, labels)))

    # Splitting the data into training and testing for each class
    train_data_class1, test_data_class1 = train_test_split(data_class1, test_size=test_size, random_state=42)
    train_data_class2, test_data_class2 = train_test_split(data_class2, test_size=test_size, random_state=42)

    # Combining training and testing data for both classes
    train_data = train_data_class1 + train_data_class2
    test_data = test_data_class1 + test_data_class2

    shuffle(train_data)
    shuffle(test_data)

    return train_data, test_data

def get_data_arrays(nested_list, img_h, img_w):
    """
    Returns array of image that can be used in a convolutional network.

    Parameters
    ----------
    nested_list : nested list of image arrays with corresponding class labels.
    img_h : Image height.
    img_w : Image width.

    Returns
    -------
    img_arrays : Numpy array
        4D Array with the size of (n_data, img_h, img_w, 1)
    label_arrays : Numpy array
        1D array with the size (n_data).
    """
    img_arrays = np.zeros((len(nested_list), img_h, img_w), dtype=np.float32)
    label_arrays = np.zeros((len(nested_list)), dtype=np.int32)
    for ind in range(len(nested_list)):
        img_arrays[ind] = nested_list[ind][0]  # accesses the file path
        label_arrays[ind] = nested_list[ind][1]  # accesses the label
    img_arrays = np.expand_dims(img_arrays, axis=3)  # increases the number of dimensions
    return img_arrays, label_arrays

def get_train_test_arrays_separate(data_path_class1, data_path_class2, img_h, img_w, test_size=0.2):
    """
    Loads training and testing data from separate directories for each class.

    Parameters
    ----------
    data_path_class1 : Path to the directory for class 1.
    data_path_class2 : Path to the directory for class 2.
    img_h : Image height.
    img_w : Image width.
    test_size : Proportion of the dataset to include in the test split.

    Returns
    -------
    train_img : Numpy array
        Training image array.
    test_img : Numpy array
        Testing image array.
    train_label : Numpy array
        Training label array.
    test_label : Numpy array
        Testing label array.
    """
    train_data, test_data = get_data_separate(data_path_class1, data_path_class2, img_h, img_w, test_size)

    train_img, train_label = get_data_arrays(train_data, img_h, img_w)
    test_img, test_label = get_data_arrays(test_data, img_h, img_w)

    return train_img, test_img, train_label, test_label
