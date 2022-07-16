import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from resnet_cnnsovnet_dynamic_routing_entropy import *
from constants import *
from utils import *
from data_loaders import *
from smallNorb import *

best_accuracy = 0.0
num_epochs = DEFAULT_EPOCHS
loss_criterion = CrossEntropyLoss()
model = ResnetCnnsovnetDynamicRouting().to(DEVICE)
optimizer = optim.Adam(model.parameters(),lr=LR) # 0.001
#optimizer = optim.SGD(model.parameters(),lr=LR)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
'''

trainloader, validloader, testloader = load_small_norb(DEFAULT_BATCH_SIZE)
print(f"len(trainloader) : {len(trainloader)} | len(validloader) : {len(validloader)} | len(testloader) : {len(testloader)}")
def train(epoch):
    print('\nTrain Epoch: %d' % epoch)
    model.train()
    train_loss = 0.0
    correct = 0.0
    total = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs, entr_reg = model(inputs)
        loss = 0.8*loss_criterion(outputs, targets) + 0.2*entr_reg
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx%100 == 0:
           progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                       % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    scheduler.step()

def valid(epoch):
    print('\nValid Epoch: %d' % epoch)
    global best_accuracy
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, entr_reg = model(inputs)
            loss = 0.8*loss_criterion(outputs, targets) + 0.2*entr_reg

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*float(correct)/total
    if acc > best_accuracy:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),  
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, '../checkpoints/ResnetCnnsovnetDynamicRouting/chkpoint_entr_adam_0.2_0.001_250_run1.pth')
        best_accuracy = acc

def test():
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    print('\n\n\nTesting...\n')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, entr_reg = model(inputs)
            loss = 0.8*loss_criterion(outputs, targets) + 0.2*entr_reg

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in range(num_epochs):
    train(epoch)
    valid(epoch)

test()
