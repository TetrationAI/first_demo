#imports
import os
import demo_functions as demo_functions
import matplotlib.pyplot as plt
import numpy as np

"""load dataset from https://www.kaggle.com/datasets/alxmamaev/flowers-recognition"""
#Change the working directory to the 'Tetration' directory
#os.chdir(r"path/to/Tetration")

#paths to datasets 
folder_path_daisy = "dataset/flowers/daisy"
folder_path_sunflower = "dataset/flowers/sunflower"

#load images
daisy_images = demo_functions.read_images(folder_path_daisy)
sunflower_images = demo_functions.read_images(folder_path_sunflower)

#Split into training and testing set.
train_img, test_img, train_label, test_label = demo_functions.get_train_test_arrays_separate(folder_path_daisy, folder_path_sunflower, 256,256,0.3)

#Display example image
idx = np.random.randint(0, len(train_img))
plt.imshow(train_img[idx])
plt.title(f"Label: {train_img[idx]}")
plt.show()