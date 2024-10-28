from main import main
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
import numpy as np

def load_and_preprocess_cifar10_classes(n, x_classes, new_size):
    # Load CIFAR-10 dataset
    (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
    
    # Select x classes
    selected_classes = x_classes  # e.g., [0, 1] for airplane and automobile
    
    # Filter out images for selected classes
    mask = np.isin(train_labels, selected_classes).flatten()
    filtered_images = train_images[mask]
    filtered_labels = train_labels[mask]
    
    # Sample n images per selected class and resize
    sampled_images = []
    sampled_labels = []
    for cls in selected_classes:
        
        class_images = filtered_images[filtered_labels.flatten() == cls]
        
        # Ensure random sampling of images
        np.random.shuffle(class_images)
        sampled_cls_images = class_images[:n]
        
        # Resize images
        for img in sampled_cls_images:
            resized_img = tf.image.resize(img, [new_size, new_size]).numpy()
            sampled_images.append(resized_img)
        
        sampled_labels += [cls] * n
    
    return np.array(sampled_images), np.array(sampled_labels)

# Example usage: Load 5 images each for classes 0 (airplane) and 1 (automobile), resized to 64x64 pixels
new_size = 64
n = 5
x_classes = [0, 1]
images, labels = load_and_preprocess_cifar10_classes(n, x_classes, new_size)

# images and labels are now in-memory numpy arrays that you can use to train your CNN
print(images)