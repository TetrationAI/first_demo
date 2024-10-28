import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

alexnet_model = models.alexnet(pretrained=True)

conv_layers = alexnet_model.features

# Enable further training on pre-tranied conv weights
for param in conv_layers.parameters():
    param.requires_grad = True

classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(256 * 3 * 3, 2048), 
    nn.LeakyReLU(),
    nn.Dropout(0.5),
    nn.Linear(2048, 2048),
    nn.LeakyReLU(),
    nn.Dropout(0.5),
    nn.Linear(2048, 5), 
    nn.LogSoftmax(dim=1)
)

AlexNet = nn.Sequential(
    conv_layers,
    classifier
)