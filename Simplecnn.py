# Basic CNN for MNIST Classification implemented using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------
# Define a Simple CNN
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1 input channel, 8 filters / For Cifar, (3, 16)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 16 filters / For Cifar, (16, 32)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)  # Fully connected layer / For Cifar, Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(64, 10)          # 10 classes (digits) / For Cifar, (128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # Conv1 + ReLU
        x = F.max_pool2d(x, 2)         # Pooling (reduces size)
        x = F.relu(self.conv2(x))      # Conv2 + ReLU
        x = F.max_pool2d(x, 2)         # Pooling
        x = torch.flatten(x, 1)        # Flatten for FC
        x = F.relu(self.fc1(x))        # Fully connected
        x = self.fc2(x)                # Output layer
        return x

# ------------------------------
# Training Function
# ------------------------------
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

# ------------------------------
# Testing Function
# ------------------------------
def test(model, device, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {100. * correct / len(test_loader.dataset):.2f}%\n")

# ------------------------------
# Main Function
# ------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform: convert to tensor + normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))    #For Cifar, Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)    #.CIFAR10
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)                   #.CIFAR10

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model, optimizer
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 3 epochs (fast for lab demo)
    for epoch in range(1, 4):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == "__main__":
    main()


# Basic CNN for MNIST Classification implemented using Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Load and preprocess MNIST data
# -----------------------------
mnist = tf.keras.datasets.mnist        #cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = mnist.load_data()    #cifar10.load_data()

# Normalize and reshape (MNIST images are 28x28 grayscale)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0    #For Cifar, x_train = x_train/255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0     #x_test = x_test/255.0

# -----------------------------
# Build a Simple CNN model
# -----------------------------
model = Sequential([
    # Convolution layer (extracts features)
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),    #For Cifar, input_shape=(32,32,3)
    MaxPooling2D((2, 2)),
    
    # Second convolution layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten and fully connected layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # helps prevent overfitting
    Dense(10, activation='softmax')  # output layer for 10 classes
])

# -----------------------------
# Compile the model
# -----------------------------
model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=[SparseCategoricalAccuracy()]
)

# -----------------------------
# Train the model
# -----------------------------
model.fit(x_train, y_train, epochs=2, batch_size=64, verbose=1)

# -----------------------------
# Evaluate the model
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
