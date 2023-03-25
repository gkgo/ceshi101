# import torch
# import torchvision.transforms as transforms
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torchvision
# import time
# from tqdm import tqdm
# from res import SAM
# from res import WideResNet


# # Set device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Load data
# train_data = torchvision.datasets.CIFAR100('./',
#                     transform=transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
# ]), download=True)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True,num_workers=0)
# val_data = torchvision.datasets.CIFAR100('./',transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
# ]),  train=False, download=True)
# val_loader = torch.utils.data.DataLoader(val_data,batch_size=128, shuffle=False, num_workers=0)


# # model = ConvNet4(num_classes=len(train_data.classes)).to(device)
# # model = resnet18gai().to(device)


# # Initialize model with SAM optimizer
# model = WideResNet(depth=16, width_factor=8, dropout=0.0, in_channels=3, labels=100).to(device)
# base_optimizer = optim.SGD
# optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9,weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.base_optimizer, milestones=[30,40], gamma=0.05)
# # Define loss function
# criterion = nn.CrossEntropyLoss()

# # Train the model
# num_epochs = 50
# best_acc = 0.0
# for epoch in range(num_epochs):
#     train_loss = 0.0
#     train_correct = 0
#     train_total = 0
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0
#     start_time = time.time()

#     # Train
#     model.train()
#     with tqdm(train_loader, unit="batch") as tepoch:
#         tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
#         for inputs, labels in tepoch:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             # First forward-backward pass
#             loss.backward()
#             optimizer.first_step(zero_grad=True)

#             # Second forward-backward pass
#             criterion(model(inputs), labels).backward()
#             optimizer.second_step(zero_grad=True)

#             train_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()
#             tepoch.set_postfix(loss=train_loss/train_total, acc=train_correct/train_total)

#     # Validate
#     model.eval()
#     with torch.no_grad():
#         with tqdm(val_loader, unit="batch") as tepoch:
#             tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
#             for inputs, labels in tepoch:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
#                 tepoch.set_postfix(loss=val_loss/val_total, acc=val_correct/val_total)
#                 val_acc = val_correct / val_total


#     lr_scheduler.step()
#     end_time = time.time()
#     epoch_time = time.time() - start_time
#     print(f'[ log ] roughly {(num_epochs - epoch) / 3600. * epoch_time:.2f} h left\n')
#     print(f"Train loss: {train_loss/train_total:.4f}, Train accuracy: {train_correct/train_total:.4f}")
#     print(f"Validation loss: {val_loss/val_total:.4f}, Validation accuracy: {val_acc:.4f}")
#     # Update best accuracy
#     if val_acc > best_acc:
#         best_acc = val_acc
#         # torch.save(model.state_dict(), save_path)

# print(f"Best validation accuracy: {best_acc:.4f}")
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
from res import WideResNet
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

    train_data = torchvision.datasets.CIFAR100('./',
                                               transform=transforms.Compose([
                                                   transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomRotation(15),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                        (0.2675, 0.2565, 0.2761)),
                                               ]), download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    val_data = torchvision.datasets.CIFAR100('./', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]), train=False, download=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=0)

    model = WideResNet(depth=16, width_factor=8, dropout=0.0, in_channels=3, labels=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.05)
    best_acc = 0.0

    for epoch in range(50):
        start_time = time.time()
        train(epoch, model, device, train_loader, optimizer)
        test_acc = test(model, device, val_loader)
        lr_scheduler.step()
        epoch_time = time.time() - start_time
        if test_acc > best_acc:
            best_acc = test_acc
        print(f'[ log ] roughly {(50 - epoch) / 3600. * epoch_time:.2f} h left\n')
        print(f"Best Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()



