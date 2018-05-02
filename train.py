import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np

import math
import os

from models import *

# TODO: make main function
# TODO: create argument list with argparser

# Loading data
mean_cifar_100 = (0.5071, 0.4865, 0.4409)
std_cifar_100 = (0.2673, 0.2564, 0.2762)

batch_size = 128
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_cifar_100, std_cifar_100)])

train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_cifar_100, std_cifar_100)])

val_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

# Define model for CIFAR-100
model = resnet50_sa(num_classes = 100)

# Get number of parameters
print 'Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad)

# Decide to use GPUs or not
use_cuda = torch.cuda.is_available()
if use_cuda:
    if torch.cuda.device_count() > 1:
        print "Using %d GPUs" % torch.cuda.device_count()
        model = nn.DataParallel(model)
    model.cuda()
    loss_func = nn.CrossEntropyLoss().cuda()    # Loss function
else:
    loss_func = nn.CrossEntropyLoss()

init_lr = 0.1
optimizer = optim.SGD(model.parameters(), lr = init_lr, momentum = 0.9, weight_decay = 5e-4)

# Training parameters
num_epochs = 200

def adjust_learning_rate(optimizer, lr):
    lr_tmp = lr * 0.2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_tmp
    return lr_tmp

# Create folder to store trained parameters
store_path = '../trained_model'
if not os.path.exists(store_path):
    os.mkdir(store_path)

epoch_train_loss_sum = []
epoch_test_loss_sum = []
lr_curr = init_lr
best_val_acc = -1

for epoch in range(num_epochs): # From 0 to num_epochs-1
    # Training
    model.train()
    # Adjust learning rate
    if epoch in [60, 120, 160]:
        lr_curr = adjust_learning_rate(optimizer, lr_curr)
    train_running_loss = 0.0
    for i, data in enumerate(train_loader):
        # Get the inputs
        images, labels = data
        # Convert inputs to variables
        if use_cuda:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)
        # Zero out the gradients of network parameters
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Loss function
        loss = loss_func(outputs, labels)
        train_running_loss += loss.data[0]
        # Backward pass
        loss.backward()
        # Update parameter
        optimizer.step()
        # Print loss every 200 minibatches
        if i % 100 == 99 or ((i+1) == train_loader.__len__()):
            print 'Epoch [%d/%d] Step [%d/%d], lr: %f, Loss: %.4f' % (epoch + 1,
                num_epochs, i+1, train_loader.__len__(), lr_curr, loss.data[0])
    epoch_train_loss_sum += [train_running_loss]

    # Validating
    model.eval()
    correct, total = 0, 0
    test_running_loss = 0
    for data in val_loader:
        images, labels = data
        if use_cuda:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images, labels)
        # Forward pass
        outputs = model(images)
        # Loss
        loss = loss_func(outputs, labels)
        test_running_loss += loss.data[0]
        # Prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    epoch_test_loss_sum += [test_running_loss]
    # Evaluate accuracy
    val_acc = 100.0 * correct / total
    print 'Epoch [%d/%d] Validation accuracy is %f %%' % (epoch + 1, num_epochs, val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print 'Saving best model at epoch %d' % (epoch + 1)
        torch.save(model.state_dict(), os.path.join(store_path, 'best_model.pkl'))

print 'Best validation accuracy is %f' % (best_val_acc)

# Save all losses value
print 'Saving all loss values to files'
np.savetxt(os.path.join(store_path, 'epoch_train_loss_sum.txt'), np.array(epoch_train_loss_sum), fmt='%f')
np.savetxt(os.path.join(store_path, 'epoch_test_loss_sum.txt'), np.array(epoch_test_loss_sum), fmt='%f')
