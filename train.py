import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from torchvision.datasets import ImageFolder
from tqdm import tqdm  # 添加进度条库
from resnet import resnet12


GPU = torch.cuda.is_available()



def train(epoch, model, device, train_loader, optimizer):
    model.train()
    size = len(train_loader.dataset)
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(
            tqdm(train_loader, desc=f"Training epoch {epoch}", unit="batch", leave=True)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += output.argmax(dim=1).eq(target).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / size
    print(f"Train Epoch: {epoch}\tLoss: {train_loss:.4f}\tAccuracy: {train_acc:.2f}%")



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing", unit="batch", leave=True)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()
            test_correct += output.argmax(dim=1).eq(target).sum().item()
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}\tAccuracy: {test_acc:.2f}%")
    return test_acc

def main():
    device = torch.device("cuda" if GPU else "cpu")


    train_data = torchvision.datasets.CIFAR10('./',
                                               transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]), download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    val_data = torchvision.datasets.CIFAR10('./', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]), train=False, download=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=0)
    # model = torchvision.models.resnet18(pretrained=True)
#     model = ConvNet4(num_classes=10).to(device)
    # model = UPANets(16, 100, 1, 32).to(device)
    model = resnet12().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 70], gamma=0.05)
    best_acc = 0.0

    for epoch in range(80):
        start_time = time.time()
        train(epoch, model, device, train_loader, optimizer)
        test_acc = test(model, device, val_loader)
        lr_scheduler.step()
        epoch_time = time.time() - start_time
        if test_acc > best_acc:
            best_acc = test_acc
        print(f'[ log ] roughly {(80 - epoch) / 3600. * epoch_time:.2f} h left')
        print(f"Best Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()

