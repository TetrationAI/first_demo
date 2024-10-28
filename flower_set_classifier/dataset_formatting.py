import os
import shutil
from sklearn.model_selection import train_test_split

os.chdir("/mnt/c/Users/Gästkonto/Documents/Programmering/projekt/TetrationAI")

# Check if it worked
print("Current Working Directory:", os.getcwd())
# gives the dataset the correct structure: 
"""
Source Data 
    - class 1 
    - class 2 
    - class 3 
    - class ...  
"""

def dataset_formatting(): 
    source_dir_path = 'datasets/flowers' 
    destination_dir_path = "datasets/flowers_mod"

    training_subdir = os.path.join(destination_dir_path, 'train')
    testing_subdir = os.path.join(destination_dir_path, 'test')

    os.makedirs(destination_dir_path, exist_ok=True)
    os.makedirs(training_subdir, exist_ok=True)
    os.makedirs(testing_subdir, exist_ok=True)

    image_dict = {}

    for subdir in os.listdir(source_dir_path):
        subdir_path = os.path.join(source_dir_path, subdir)

        if not os.path.isdir(subdir_path):  # ignores .DS_Store
            continue
        
        # Create subdirectories for training and testing images separately
        destination_subdir_train = os.path.join(training_subdir, subdir)
        os.makedirs(destination_subdir_train, exist_ok=True)

        destination_subdir_test = os.path.join(testing_subdir, subdir)
        os.makedirs(destination_subdir_test, exist_ok=True)

        image_dict[subdir] = []

        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)

            if os.path.isfile(file_path) and not file_name.startswith('.'):
                image_dict[subdir].append(file_path)

    for key in image_dict.keys():
        train_paths, test_paths = train_test_split(image_dict[key], test_size=0.1)

        for file_path in train_paths:
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(training_subdir, key, file_name)
            shutil.copy(file_path, destination_path)

        for file_path in test_paths:
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(testing_subdir, key, file_name)
            shutil.copy(file_path, destination_path)

if __name__ == "__main__":
    dataset_formatting()
