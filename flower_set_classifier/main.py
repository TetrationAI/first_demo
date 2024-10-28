import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from partially_pretrained_architecture import gen_alexnet
from utils import get_train_valid_loader, get_test_loader, show_images
import matplotlib.pyplot as plt
import torchvision.utils as vutils 
import os 

# Set the working directory
os.chdir("/mnt/c/Users/Gästkonto/Documents/Programmering/projekt/TetrationAI")

# Check if it worked
print("Current Working Directory:", os.getcwd())

def main(train_valid_data_path = 'datasets/flowers', 
         test_data_path = None, 
         model_name = "biggan_discriminator_30_own_dataset", 
         num_original_classes = 5, 
         discriminator_save_dir = "classifiers/"):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"device: {device}")

    save_model_metrics_path = os.path.join(discriminator_save_dir, f'{model_name}_model_metrics_V2.pth')
    save_model_dict_path = os.path.join(discriminator_save_dir, f'{model_name}_state_dictionary.pth')
    save_full_model_path = os.path.join(discriminator_save_dir, f'{model_name}_full_model.pth')
    
    num_original_classes = num_original_classes

    # total number of classes 
    num_classes = 30

    size = num_classes-num_original_classes

    x_classes = list(np.random.choice(range(0, num_classes), size=size, replace=False))

    num_epochs = 500
    batch_size = 64 # 128 
    learning_rate = 0.001
    image_size = 128 # 128
    seed = 69

    #early stopping
    patience = 30 
    best_val_loss = float('inf')
    current_patience = 0

    validation_accuracies = []
    testing_accuracies = []
    training_accuracies = []
    training_losses = []
    validation_losses = []

    model = gen_alexnet(num_classes)
    #model = simple_alexnet()

    model = model.to(device)

    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # loss, optimizer, lr schedule
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001) # weight_decay = l2 regularizaton 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=8)
    
    # dataloaders
    # This dataloader should be fed with 
    train_loader, valid_loader, class_map  = get_train_valid_loader(
                        data_dir= train_valid_data_path, # CHANGE THIS DATA DIRECTORY SO THAT A NEW IS CREATED FOR EACH TIME. 
                        split = 0.2, 
                        batch_size=batch_size,
                        augment=True,
                        random_seed=1, 
                        size = image_size, 
                        imagenet=True, 
                        num_classes = num_original_classes, 
                        x_classes = x_classes 
                        )

    print("IN MAIN", class_map)
    test_loader = False
    if test_loader:  
        test_loader = get_test_loader(
                            data_dir=test_data_path,
                            batch_size=batch_size, 
                            size = image_size,
                            imagenet= True)

    # Train
    total_step_train = len(train_loader)
  
    for epoch in range(num_epochs):
        model.train() 
        loss_sum = 0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # plot images for the first time 
            if epoch == 0 and i == 0:  
                show_images(images, num_images=16, nrow=4, normalize=True)
                print(labels)
            
            # forward pass
            outputs = model(images)

            loss = criterion(outputs, labels)

            loss_sum += loss.item()

            # training labels 
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_acc = correct / total * 100

        training_accuracies.append(train_acc)

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Accuracy: {:.4f}%' 
                    .format(epoch+1, num_epochs, i+1, total_step_train, loss_sum, train_acc))
        
        #save losses
        average_train_loss = loss_sum / total_step_train
        training_losses.append(average_train_loss)

        # Validation
        model.eval() 
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                # calculate validation loss to update lr schedule 
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print('Accuracy of the network on the {} validation images: {:.4f} %'.format(total, val_accuracy))
            print(" ")
            validation_accuracies.append(val_accuracy)
            average_val_loss = val_loss / len(valid_loader)
            validation_losses.append(average_val_loss)

        # check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_patience = 0
            best_model_dict = model.state_dict()
        else:
            current_patience += 1

        # check for early stopping
        if current_patience >= patience:
            print(f'Early stopping after {epoch} epochs.')
            break

        # update lr schedule 
        scheduler.step(val_loss / len(valid_loader))

        #if epoch % 5 == 0: 
            # Save model that's noe compatible with biggan-am,
            #torch.save(best_model_dict, f'pul_nod_model_ep{epoch}_DICT.pth')
            #torch.save(model, f'pul_nod_model_ep{epoch}_DICT.pth')
    
    # Save model that's noe compatible with biggan-am,
    torch.save(best_model_dict, save_model_dict_path)
    torch.save(model, save_full_model_path)

    #Test model
    if test_loader: 
        model.load_state_dict(best_model_dict)
        model.eval() # Check if this is correct
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            test_accuracy = 100 * correct / total
            testing_accuracies.append(test_accuracy)

            print('Accuracy of the network on the {} test images: {} %'.format(total, test_accuracy))

    torch.save({
    'training_accuracies': training_accuracies, 
    'validation_accuracies': validation_accuracies,
    'training_losses': training_losses, 
    'validation_losses': validation_losses,
    "class_map": class_map
    }, save_model_metrics_path)

    # Plot metrics
    plt.plot(range(1, num_epochs + 1), validation_accuracies, label='Validation Accuracy')
    plt.plot(range(1, num_epochs + 1), training_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()