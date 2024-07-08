import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from model import resnet50_scratch_dag
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from train import train_epoch, val_epoch, train_model, train_val_split

# Importing PyTorch libraries and modules for deep learning operations
import torchvision

import numpy as np
# Function to set the gradient requirement of the model parameters
def set_parameter_requires_grad(model, feature_extracting, num_layers):
    for i, param in enumerate(model.parameters()):
        param.requires_grad = i >= (len(list(model.parameters())) - num_layers) if feature_extracting else True

def main():
    # Directly using fixed values instead of args
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    num_epochs = 50
    save_path = "../checkpoints/result"
    num_classes = 10
    arch = "resnet50_scratch_dag"

    model = resnet50_scratch_dag(weights_path='./models/resnet50_scratch_dag.pth')
    set_parameter_requires_grad(model, True, 2)  # Only last 2 layers are trainable

    # Define transformations
    train_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))
    ])
    train_dataset = torchvision.datasets.ImageFolder('./res_aligned/', transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder('./res_aligned/', transform=val_transforms)
    # Assume train_dataset and val_dataset are defined somewhere
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    criterion = losses.ArcFaceLoss(num_classes=2000, embedding_size=2048, margin=35, scale=64)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion_optimizer  = torch.optim.SGD(criterion.parameters(), lr = 0.01, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(model, train_loader, val_loader, criterion, optimizer,criterion_optimizer, lr_scheduler, num_epochs, save_path, arch)

if __name__ == '__main__':
    main()
