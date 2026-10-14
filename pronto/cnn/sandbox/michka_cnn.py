import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# !pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# !pip install torchmetrics:
import torchmetrics

batch_size = 60

train_dataset = datasets.MNIST(root="pronto/Mnist_data/", download=True, train=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="pronto/Mnist_data/", download=True, train=False, transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

"""
# TO verify by visualisation that the datasets were correctly fitted into DataLoader type objects and to
# assert that the data in the datasets are valid

def imshow(img):
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
# show images
imshow(torchvision.utils.make_grid(images))

""" 

class CNN(nn.Module):
   def __init__(self, in_channels, num_classes):

       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super(CNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       # Max pooling layer
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       return x
   

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=1, num_classes=10).to(device)
""" # To verufy it's the rigtht model
print(model)

# The output of this print should be : 
CNN(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (fc1): Linear(in_features=784, out_features=10, bias=True)
)
# and it is !!
"""

def training(model,device):
    """Trains the model withoutwatermaking it, using 115 batches from the Minist dataset"""
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs=115 # The minimum ideal number of epoch for training is around 115 epochs according to the paper 
    for epoch in range(num_epochs):
    # Iterate over training batches
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def acc_calculation(model : CNN) :
    '''
    renvoie la précision d'un modèle
    ''' 
    # Set up of multiclass accuracy metric
    acc = torchmetrics.Accuracy(task="multiclass",num_classes=10)

    # Iterate over the dataset batches
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # Get predicted probabilities for test data batch
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            acc(preds, labels)

    #Compute total test accuracy
    test_accuracy = acc.compute()
#    print(f"Test accuracy: {test_accuracy}")
    return test_accuracy


#Signature à insérer 
watermark_size = 200
torch.manual_seed(42)
B = torch.randint(0, 2, (watermark_size))

def direct(model : CNN, Xdirect:torch.Tensor, device)->CNN:
    '''
    Training modèle de CNN avec une méthode de signature Xdirect (un 1 par ligne
    que des 0 sinon)
    '''
    model = copy.deepcopy(model)
    # Define the loss function ()
    a = 1e-4
    W_mean = model.conv1.weight.mean(dim=0)  # On met la signature dans la couche 1 
    w = torch.flatten(W_mean)
    Y = torch.sigmoid(w@Xdirect)
    criterion = nn.CrossEntropyLoss() + a*F.binary_cross_entropy(Y, B)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs=115 # The minimum ideal number of epoch for training is around 115 epochs according to the paper 
    for epoch in range(num_epochs):
    # Iterate over training batches
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def diff(model:CNN, Xdiff:torch.Tensor, device)->CNN:
    '''
    Training modèle de CNN avec une méthode de signature Xdiff (un 1 et un -1 par ligne
    que des 0 sinon)
    '''
    model = copy.deepcopy(model)
    # Define the loss function
    # Define the loss function ()
    a = 1e-4
    W_mean = model.conv1.weight.mean(dim=0)  # On met la signature dans la couche 1 
    w = torch.flatten(W_mean)
    Y = torch.sigmoid(w@Xdiff)
    criterion = nn.CrossEntropyLoss()+ a*F.binary_cross_entropy(Y, B)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs=115 # The minimum ideal number of epoch for training is around 115 epochs according to the paper 
    for epoch in range(num_epochs):
    # Iterate over training batches
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def random(model:CNN, Xrandom:torch.Tensor, device)->CNN:
    '''
    Training modèle de CNN avec une méthode de signature Xdiff (un 1 et un -1 par ligne
    que des 0 sinon)
    '''
    model = copy.deepcopy(model)
    # Define the loss function
    # Define the loss function ()
    a = 1e-4
    W_mean = model.conv3.weight.mean(dim=0)  # On met la signature dans la couche 1 
    w = torch.flatten(W_mean)
    Y = torch.sigmoid(w@Xrandom)
    criterion = nn.CrossEntropyLoss()+ a*F.binary_cross_entropy(Y, B)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs=115 # The minimum ideal number of epoch for training is around 115 epochs according to the paper 
    for epoch in range(num_epochs):
    # Iterate over training batches
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
