import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

# ========== Toggle Device ==========
use_gpu = True  # Set to False for CPU
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ===================================

# Set random seeds
np.random.seed(0)
torch.manual_seed(0)

# Load MNIST dataset
training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)

STARTSIZE = 8

# Model definition
class LargeConvNet(nn.Module):
    def __init__(self):
        super(LargeConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, STARTSIZE, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(STARTSIZE, STARTSIZE*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(STARTSIZE*2, STARTSIZE*4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        # After 2 poolings: 28x28 -> 14x14 -> 7x7 = 128 * 7 * 7
        self.fc1 = nn.Linear(STARTSIZE*4 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))  # No third pooling
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Training function
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")


# Test function
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    print(f"Test Error: \n Accuracy: {100 * accuracy:.1f}%, Avg loss: {test_loss:.4f}\n")


# Initialize model, optimizer, loss
model = LargeConvNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

import time

x = time.perf_counter()
# Training loop
epochs = 2
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_loop(test_dataloader, model, loss_fn, device)

print("Done!")
print(time.perf_counter() - x)