import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.utils.data import ConcatDataset
import os 

import tensorflow as tf

from torch.utils.data import Dataset 
import matplotlib.pyplot as plt
import torchvision.utils as vutils 

from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

def load_and_preprocess_cifar10_classes(n, x_classes, new_size):
    # load the dataset 
    (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()

    # Select x classes
    selected_classes = x_classes  # e.g., [0, 1] for airplane and automobile
    
    # Filter out images for selected classes
    mask = np.isin(train_labels, selected_classes).flatten()
    filtered_images = train_images[mask]
    filtered_labels = train_labels[mask]
    
    # sample the desired number of images per selected class and resize
    sampled_images = []
    sampled_labels = []
    for cls in selected_classes:
        
        class_images = filtered_images[filtered_labels.flatten() == cls]
        
        # random sampling of images
        np.random.shuffle(class_images)
        sampled_cls_images = class_images[:int(n)]
        
        # resize images
        for img in sampled_cls_images:
            resized_img = tf.image.resize(img, [new_size, new_size]).numpy()
            sampled_images.append(resized_img)
            sampled_labels.append(cls)

    return np.array(sampled_images), np.array(sampled_labels)

# Complement with images from the cifar10 dataset. 
def load_and_preprocess_cifar100_classes(n, x_classes, new_size, label_mode='fine'):
    
    cifar100_classes = {
        0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
        5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
        10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
        15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
        20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
        25: "couch", 26: "cra", 27: "crocodile", 28: "cup", 29: "dinosaur",
        30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
        35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
        40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
        45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
        50: "mouse", 51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid",
        55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree",
        60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
        65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
        70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
        75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
        80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table",
        85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
        90: "train", 91: "trout", 92: "tulip", 93: "turtle", 94: "wardrobe",
        95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm"
    }
    result_dict = {cifar100_classes[index]: position for position, index in enumerate(x_classes)}
    
    # load the dataset
    (train_images, train_labels), _ = tf.keras.datasets.cifar100.load_data(label_mode=label_mode)

    selected_classes = x_classes
    
    # create mapping from original class indices to new indices (starting from 0)
    class_map = {original: idx for idx, original in enumerate(selected_classes)}

    # Filter out images for selected classes
    mask = np.isin(train_labels.flatten(), selected_classes)
    filtered_images = train_images[mask]
    filtered_labels = train_labels[mask]
    
    # Map the original class labels to the new "label system"
    mapped_labels = np.array([class_map[label[0]] for label in filtered_labels])

    # sample the desired number of images per selected class and resize
    sampled_images = []
    sampled_labels = []
    for new_cls in range(len(selected_classes)):
        class_images = filtered_images[mapped_labels == new_cls]
        
        # random sampling of images
        np.random.shuffle(class_images)
        sampled_cls_images = class_images[:int(n)]
        
        # Resize images
        for img in sampled_cls_images:
            resized_img = tf.image.resize(img, [new_size, new_size]).numpy()
            sampled_images.append(resized_img)
            sampled_labels.append(new_cls)

    return np.array(sampled_images), np.array(sampled_labels), result_dict

# ensure labels and images are of same datatypes 
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])  
    return images, labels

# define a vustom dataset 
class CustomTensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            from torchvision.transforms.functional import to_pil_image
            image = to_pil_image(image)  
            image = self.transform(image)

        return image, label
    
    def get_labels(self):
        return self.labels

class LightingNoise(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
    
# displays a batch of images 
def show_images(images, num_images=16, nrow=4, normalize=True):
    
    # Make a grid from the batch
    images = images.detach().cpu()  # Move images to CPU and detach from the computation graph
    grid = vutils.make_grid(images[:num_images], nrow=nrow, padding=2, normalize=normalize)

    plt.figure(figsize=(nrow*3, num_images/nrow*3))  # Adjust the size of the figure
    plt.imshow(grid.permute(1, 2, 0))  # Permute the tensor to image format for display
    plt.axis('off')  # Turn off the axis
    plt.show()

def calculate_mean_std(data_dir, size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def get_train_valid_loader_OLD(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           size, 
                           valid_size=0.1,
                           shuffle=True, 
                           imagenet = False):
    if imagenet:
        # ImageNet normalization values
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        # Normalize from the data itself
        mean, std = calculate_mean_std(data_dir, size, batch_size=batch_size)
    
    normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor(),
    normalize,

])

    else:
        train_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = ImageFolder(
    root=data_dir,
    transform=train_transform
    )

    valid_dataset = ImageFolder(
    root=data_dir,
    transform=valid_transform
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    batch_size,
                    size,
                    shuffle=True,
                    imagenet = False):
    if imagenet:
        # ImageNet normalization values
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean, std = calculate_mean_std(data_dir, size, batch_size=batch_size)
    
    normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    # define transform
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = ImageFolder(
    root=data_dir,
    transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

def get_train_valid_loader(data_dir, split, batch_size, augment, random_seed, size, x_classes, shuffle=True, imagenet=False, num_classes=10):
    
    if imagenet:
        # ImageNet normalization values
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean, std = calculate_mean_std(data_dir, size, batch_size=batch_size)
    
    normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    
    # Transformations...
    valid_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor(),
    normalize,

    ])

    else:
        train_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            normalize,
        ])
    
    # calculate number of images to be generated 
    num_images = 0

    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        num_images += len(os.listdir(subdir_path))
    

    split_ratio = 1 - split 

    # Calculate the sizes of the splits
    num_train_original = int(num_images * split_ratio)
    num_valid_original = num_images - num_train_original

    # load data
    dataset = ImageFolder(root=data_dir, transform=train_transform)
    original_class_map = dataset.class_to_idx
    # Split the dataset into training and validation datasets
    train_dataset, valid_dataset = random_split(dataset, [num_train_original, num_valid_original])
   
    
    # debugging
    print("original train shape (train_dataset)", len(train_dataset))
    print("original valid shape (valid_dataset)", len(valid_dataset))
    

    # If less than the number of desired classes... 
    if len(x_classes) >= 1:
        # find the mean number of images to extract per class 
        num_images_per_class = int(num_images / len(os.listdir(data_dir)))

        print("x_classes in loader", x_classes)
        cifar_images, cifar_labels, class_map = load_and_preprocess_cifar100_classes(n = num_images_per_class, 
                                                                          x_classes = x_classes, 
                                                                          new_size = size)
        print("CIFAR labels", np.min(cifar_labels), np.max(cifar_labels))
        print("Images shape", cifar_images.shape)
        print("Labels shape", cifar_labels.shape)

        N = len(os.listdir(data_dir))  # Number of classes in the original dataset
        
        # remap labels in the class map
        adjusted_class_map = {original: idx + N for original, idx in class_map.items()}
        # When loading CIFAR, remap its labels
        cifar_labels = np.array([label + N for label in cifar_labels])

        indices = range(len(cifar_labels))
        train_indices, valid_indices, _, _ = train_test_split(
            indices, cifar_labels, stratify=cifar_labels, test_size=(1 - split_ratio), random_state=42
        )

        # Use indices to create the actual training and validation sets
        cifar_images_train = cifar_images[train_indices]
        cifar_images_val = cifar_images[valid_indices]
        cifar_labels_train = cifar_labels[train_indices]
        cifar_labels_val = cifar_labels[valid_indices]

        # Convert to tensors and normalize
        cifar_images_train = torch.tensor(cifar_images_train.transpose((0, 3, 1, 2))).float() / 255 # Convert to tensors and normalize
        cifar_images_val = torch.tensor(cifar_images_val.transpose((0, 3, 1, 2))).float() / 255

        #cifar_images_train = cifar_images_train.transpose((0, 3, 1, 2))
        #cifar_images_val = cifar_images_val.transpose((0, 3, 1, 2))
        
        
        # Convert labels to tensors 
        cifar_labels_train = torch.tensor(cifar_labels_train)
        cifar_labels_val = torch.tensor(cifar_labels_val)
        
        # debugging
        print("NEW train_images shape", cifar_images_train.shape)
        print("NEW train_labels shape", cifar_labels_train.shape)
        print("NEW valid_images shape", cifar_images_val.shape)
        print("NEW valid_labels shape", cifar_labels_val.shape)
        
        # Make the dataset a tensorflow dataset... 
        cifar_train_dataset = CustomTensorDataset(images = cifar_images_train, labels= cifar_labels_train, transform = train_transform)
        
        cifar_valid_dataset = CustomTensorDataset(images = cifar_images_val, labels= cifar_labels_val, transform = valid_transform)

        # debugging
        #print("cifar_train_dataset shape", len(cifar_train_dataset))
        #print("cifar_valid_dataset shape", len(cifar_valid_dataset))
        #print("original train visualizeation (train_dataset)", train_dataset)
        #print(" train visualizeation (train_dataset)", cifar_train_dataset)
        # Combine original dataset with CIFAR-10 dataset
        train_dataset = ConcatDataset([train_dataset, cifar_train_dataset])
        
        valid_dataset = ConcatDataset([valid_dataset, cifar_valid_dataset])

        class_map = {**original_class_map, **adjusted_class_map}
    else: 
        num_valid = int(split * num_images)
        num_train = num_images - num_valid

        #split images 
        cifar_images_train = train_dataset[:num_train]
        cifar_images_val = valid_dataset[num_train:(num_train + num_valid)]

        cifar_labels_train = cifar_labels[:num_train]
        cifar_labels_val = cifar_labels[num_train:(num_train + num_valid)]

        class_map = original_class_map
   
    print("shape train dataset", len(train_dataset))
    print("shape valid dataset", len(valid_dataset))

    num_train = len(train_dataset)
    num_valid = len(valid_dataset)
    train_idx = list(range(num_train))
    valid_idx =  list(range(num_valid))
    
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(train_idx)
        np.random.shuffle(valid_idx)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=custom_collate_fn)
    
    return train_loader, valid_loader, class_map

def get_loader_for_fold(data_dir, indices, batch_size, augment, size, imagenet=False):
    if imagenet:
        # ImageNet normalization values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = calculate_mean_std(data_dir, size, batch_size=batch_size)
    
    normalize = transforms.Normalize(mean=mean, std=std)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    if augment:
        train_transform.transforms.insert(0, transforms.RandomHorizontalFlip())
        train_transform.transforms.insert(0, transforms.RandomRotation(10))
        # Add other augmentations here as needed

    # Load the dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # Creating data loaders for the specified fold
    train_sampler = SubsetRandomSampler(indices['train'])
    valid_sampler = SubsetRandomSampler(indices['valid'])

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader



