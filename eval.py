# Loading test data
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

def evaluate(args):
    mean_cifar_100 = (0.5071, 0.4865, 0.4409)
    std_cifar_100 = (0.2673, 0.2564, 0.2762)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_cifar_100, std_cifar_100)])

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=8)

    # Define model for CIFAR-100
    model = get_model(args.architecture)
    print 'Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if torch.cuda.device_count() > 1:
            print "Using %d GPUs" % torch.cuda.device_count()
            model = nn.DataParallel(model)
        model.cuda()

    # Load trained parameters
    trained_model = os.path.join(args.trained_model, args.architecture)
    try:
        model.load_state_dict(torch.load(os.path.join(trained_model, 'best_model.pkl')))
        print 'Trained parameters are loaded'
    except:
        raise Exception('Cannot load trained parameters')

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--architecture', type=str, default='resnet50',
                         help='Architecture to use')
    parser.add_argument('--batch_size', type=int, default=128,
                         help='Batch size')
    parser.add_argument('--trained_model', type=str, default='../trained_model',
                         help='Path to store trained model')

    args = parser.parse_args()

    print 'Hyperparameters:'
    for arg in vars(args):
        print '\t%s:\t%s' %(arg, getattr(args, arg))

    evaluate(args)
