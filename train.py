import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import time
from tqdm import tqdm
from res import SAM
from res import WideResNet


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data
train_data = torchvision.datasets.CIFAR100('./dataset',
                    transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
]), download=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True,num_workers=0)
val_data = torchvision.datasets.CIFAR100('./dataset',transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
]),  train=False, download=False)
val_loader = torch.utils.data.DataLoader(val_data,batch_size=128, shuffle=False, num_workers=0)


# model = ConvNet4(num_classes=len(train_data.classes)).to(device)
# model = resnet18gai().to(device)


# Initialize model with SAM optimizer
model = WideResNet(depth=16, width_factor=8, dropout=0.0, in_channels=3, labels=100).to(device)
base_optimizer = optim.SGD
optimizer = SAM(model.parameters(), base_optimizer,rho=2.0, lr=0.1, momentum=0.9,weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.base_optimizer, milestones=[30,40], gamma=0.05)
# Define loss function
criterion = nn.CrossEntropyLoss()

# Train the model
num_epochs = 50
best_acc = 0.0
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    start_time = time.time()

    # Train
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in tepoch:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # First forward-backward pass
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass
            criterion(model(inputs), labels).backward()
            optimizer.second_step(zero_grad=True)

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            tepoch.set_postfix(loss=train_loss/train_total, acc=train_correct/train_total)

    # Validate
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                tepoch.set_postfix(loss=val_loss/val_total, acc=val_correct/val_total)
                val_acc = val_correct / val_total


    lr_scheduler.step()
    end_time = time.time()
    epoch_time = time.time() - start_time
    print(f'[ log ] roughly {(num_epochs - epoch) / 3600. * epoch_time:.2f} h left\n')
    print(f"Train loss: {train_loss/train_total:.4f}, Train accuracy: {train_correct/train_total:.4f}")
    print(f"Validation loss: {val_loss/val_total:.4f}, Validation accuracy: {val_acc:.4f}")
    # Update best accuracy
    if val_acc > best_acc:
        best_acc = val_acc
        # torch.save(model.state_dict(), save_path)

print(f"Best validation accuracy: {best_acc:.4f}")


