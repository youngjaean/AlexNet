import torch 
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from model import *

net = alexnet()
net = net.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_acc = 0  
start_epoch = 0


transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()

best_acc = 0


def train(epoch, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    for epochs in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
            running_loss += loss.item()
            if i % 100 == 0 :    
                print('[%d, %5d] loss: %.3f' %
                    (epochs + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
        
def test(epoch, optimizer):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    

            
    acc = 100.*correct/total
    print("Accuracy :",acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Pytorch Cifar10 alexnet")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epoch', default=10, type=float)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument("--decay", default=5e-4)
    args = parser.parse_args()

    net.to(device)
    optimizer = optim.SGD(net.parameters(), 
                        lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    train(args.epoch, optimizer)
    test(args.epoch, optimizer)