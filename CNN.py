# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from torch.autograd import Variable

# Parameters
BATCH_SIZE = 100
NUM_ITERS = 2500
LEARNING_RATE = 0.001

# Load train dataset
dataset_train = pd.read_csv('mnist_train.csv', delimiter=',', header=None)
data_train = dataset_train.values
images_train, labels_train = data_train[:,1:], data_train[:, 0]

# Load test dataset
dataset_test = pd.read_csv('mnist_test.csv', delimiter=',', header=None)
data_test = dataset_test.values
images_test, labels_test = data_test[:,1:], data_test[:, 0]

# Formatting of train dataset
images_train = images_train.reshape(60000, 1, 28, 28)
images_train = torch.from_numpy(images_train).float()
labels_train = torch.from_numpy(np.array(labels_train))

# Formatting of test dataset
images_test = images_test.reshape(10000, 1, 28, 28)
images_test = torch.from_numpy(images_test).float()
labels_test = torch.from_numpy(np.array(labels_test))

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(images_train, labels_train)
test = torch.utils.data.TensorDataset(images_test, labels_test)

# Pytorh data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

# Simple feed forward convolutional neural network
class PR_CNN(nn.Module):

    def __init__(self, **kwargs):

        # Creates an CNN_basic model from the scratch
        super(PR_CNN, self).__init__()
        # Set 1
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu_1 = nn.LeakyReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        # Set 2
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu_2 = nn.LeakyReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected
        self.fc = nn.Linear(32 * 5 * 5, 10)

    # Computes forward pass on the network
    def forward(self, x):

        # Set 1
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.pool_1(out)
        # Set 2
        out = self.conv_2(out)
        out = self.relu_2(out)
        out = self.pool_2(out)
        # Flatten
        out = out.view(out.size(0), -1)
        # Fully connected
        out = self.fc(out)

        return out

# Create model
model = PR_CNN()
# Cross entropy loss
error = nn.CrossEntropyLoss()
# SGH optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
epochs = int(NUM_ITERS / (len(images_train) / BATCH_SIZE))

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and entropy loss
        loss = error(outputs, labels)
        # Calculate gradients
        loss.backward()
        # Update parameters
        optimizer.step()

        count += 1
        if count % 50 == 0:
            # Calculate accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                test = Variable(images.view(100,1,28,28))
                # Forward propagation
                outputs = model(test)
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                # Total number of labels
                total += len(labels)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / float(total)
            # Store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

# Visualization loss
plt.figure(1)
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of Iterations")

# Visualization accuracy
plt.figure(2)
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of Iterations")
plt.show()
