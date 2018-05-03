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
import argparse

from models import *

def train(args):
    """
    Main function to train the model
    """

    print '\nLoading data'

    # Loading data
    mean_cifar_100 = (0.5071, 0.4865, 0.4409)
    std_cifar_100 = (0.2673, 0.2564, 0.2762)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_cifar_100, std_cifar_100)])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                            shuffle=True, num_workers=8)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_cifar_100, std_cifar_100)])

    val_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_val)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                            shuffle=False, num_workers=8)

    # Define model for CIFAR-100
    model = get_model(args.architecture)

    # Get number of parameters
    print 'Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Decide to use GPUs or not
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if torch.cuda.device_count() > 1:
            print "Using %d GPUs" % torch.cuda.device_count()
            model = nn.DataParallel(model)
        model.cuda()
        loss_func = nn.CrossEntropyLoss().cuda()
    else:
        loss_func = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr = args.initial_lr, momentum = 0.9, weight_decay = args.weight_decay)

    # Create folder to store trained parameters
    if not os.path.exists(args.store_path):
        os.mkdir(args.store_path)

    epoch_train_loss_sum = []
    epoch_test_loss_sum = []
    lr_curr = args.initial_lr
    best_val_acc = -1

    for epoch in range(args.n_epochs):
        model.train()
        # Adjust learning rate
        if epoch in args.schedule:
            lr_curr = lr_curr * args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_curr
        train_running_loss = 0.0
        for i, data in enumerate(train_loader):
            # Get the inputs
            images, labels = data
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

            if i % 100 == 99 or ((i+1) == train_loader.__len__()):
                print 'Epoch [%d/%d] Step [%d/%d], lr: %f, Loss: %.4f' % (epoch + 1,
                    args.n_epochs, i+1, train_loader.__len__(), lr_curr, loss.data[0])

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
            outputs = model(images)
            loss = loss_func(outputs, labels)
            test_running_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        epoch_test_loss_sum += [test_running_loss]
        # Evaluate accuracy
        val_acc = 100.0 * correct / total
        print 'Epoch [%d/%d] Validation accuracy is %f %%' % (epoch + 1, args.n_epochs, val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print 'SAVING BEST MODEL AT EPOCH %d' % (epoch + 1)
            torch.save(model.state_dict(), os.path.join(args.store_path, 'best_model.pkl'))

    print 'Best validation accuracy is %f' % (best_val_acc)

    # Save all losses value
    print 'Saving all loss values to files'
    np.savetxt(os.path.join(args.store_path, 'epoch_train_loss_sum.txt'), np.array(epoch_train_loss_sum), fmt='%f')
    np.savetxt(os.path.join(args.store_path, 'epoch_test_loss_sum.txt'), np.array(epoch_test_loss_sum), fmt='%f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--architecture', type=str, default='resnet50',
                         help='Architecture to use')
    parser.add_argument('--n_epochs', type=int, default=200,
                         help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                         help='Batch size')
    parser.add_argument('--initial_lr', type=float, default=0.1,
                         help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.2,
                         help='Learning rate decay ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160],
                         help='Scheduled epoch to decrease learning rate')
    parser.add_argument('--store_path', type=str, default='../trained_model',
                         help='Path to store trained model')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                         help='Weight decay param used in loss function')

    args = parser.parse_args()

    print 'Hyperparameters:'
    for arg in vars(args):
        print '\t%s:\t%s' %(arg, getattr(args, arg))

    train(args)
