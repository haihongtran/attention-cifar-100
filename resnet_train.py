import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import math
import os
import numpy as np

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

# Define resnet for CIFAR-100
resnet = resnet50(pretrained = False, num_classes = 100)

# Decide to use GPUs or not
use_cuda = torch.cuda.is_available()
if use_cuda:
    if torch.cuda.device_count() > 1:
        print "Using %d GPUs" % torch.cuda.device_count()
        resnet = nn.DataParallel(resnet)
    resnet.cuda()
    loss_func = nn.CrossEntropyLoss().cuda()    # Loss function
else:
    loss_func = nn.CrossEntropyLoss()

init_lr = 0.1
optimizer = optim.SGD(resnet.parameters(), lr = init_lr, momentum = 0.9, weight_decay = 5e-4)

# Training parameters
num_epochs = 200

# Start training
num_epoch_decay_start = 80
num_epoch_decay_every = 10

def adjust_learning_rate(optimizer, lr):
    lr_tmp = lr * 0.2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_tmp
    return lr_tmp

# Create folder to store trained parameters
store_path = 'trained_model'
if not os.path.exists(store_path):
    os.mkdir(store_path)

epoch_train_loss_sum = []
epoch_test_loss_sum = []
lr_curr = init_lr
best_val_acc = -1

for epoch in range(num_epochs): # From 0 to num_epochs-1
    # Training
    resnet.train()
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
        outputs = resnet(images)
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
    resnet.eval()
    correct, total = 0, 0
    test_running_loss = 0
    for data in val_loader:
        images, labels = data
        if use_cuda:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images, labels)
        # Forward pass
        outputs = resnet(images)
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
        torch.save(resnet.state_dict(), os.path.join(store_path, 'best_resnet.pkl'))

print 'Best validation accuracy is %f' % (best_val_acc)

# Save all losses value
print 'Saving all loss values to files'
np.savetxt(os.path.join(store_path, 'epoch_train_loss_sum.txt'), np.array(epoch_train_loss_sum), fmt='%f')
np.savetxt(os.path.join(store_path, 'epoch_test_loss_sum.txt'), np.array(epoch_test_loss_sum), fmt='%f')
