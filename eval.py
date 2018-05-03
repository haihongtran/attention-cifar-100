# Loading test data
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from models import *

import os

# TODO: add argparser

mean_cifar_100 = (0.5071, 0.4865, 0.4409)
std_cifar_100 = (0.2673, 0.2564, 0.2762)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_cifar_100, std_cifar_100)])

batch_size = 128
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)


# Define model for CIFAR-100
model = resnet50_ca(num_classes = 100)

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    if torch.cuda.device_count() > 1:
        print "Using %d GPUs" % torch.cuda.device_count()
        model = nn.DataParallel(model)
    model.cuda()


# Load trained parameters
store_path = '../stored_model/resnet-ca/lr-decay-0.1_scheduler-60-120-160'
try:
    model.load_state_dict(torch.load(os.path.join(store_path, 'best_model.pkl')))
    print 'Trained model is loaded'
except:
    print 'Cannot load trained model'

# Evaluate trained model
correct, total = 0, 0

# Set network in evaluation mode
model.eval()

for data in testloader:
    images, labels = data
    if use_cuda:
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
    else:
        images, labels = Variable(images, labels)
    # Forward pass
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()

print 'Accuracy on %d test images is %.2f%%' % (total, 100.0 * correct/total)
