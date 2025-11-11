# LeNet Implementation in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------
# Define LeNet-5 Architecture
# ------------------------------
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # For grayscale images (MNIST) / For Cifar, (3, 6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjust depending on input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # Conv1 + ReLU
        x = F.max_pool2d(x, 2)      # Pooling
        x = F.relu(self.conv2(x))   # Conv2 + ReLU
        x = F.max_pool2d(x, 2)      # Pooling
        x = torch.flatten(x, 1)     # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)             # Output logits
        return x

# ------------------------------
# Training Setup
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
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}")

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
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({100. * correct / len(test_loader.dataset):.2f}%)\n")

# ------------------------------
# Main Function
# ------------------------------
def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST dataset (handwritten digits)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),   # LeNet expects 32x32 input
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  #For Cifar, .Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)  #.CIFAR10
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)  #.CIFAR10

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize model
    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and test
    for epoch in range(1, 6):  # Train for 5 epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()

# LeNet Implementation in Tensorflow
# LeNet Implementation in TensorFlow (Keras)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# ------------------------------
# Define LeNet-5 Architecture
# ------------------------------
def LeNet(num_classes=10):
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)),  # Conv1
        layers.AveragePooling2D(pool_size=(2, 2)),                             # Pool1
        layers.Conv2D(16, (5, 5), activation='relu'),                          # Conv2
        layers.AveragePooling2D(pool_size=(2, 2)),                             # Pool2
        layers.Flatten(),                                                      # Flatten
        layers.Dense(120, activation='relu'),                                  # FC1
        layers.Dense(84, activation='relu'),                                   # FC2
        layers.Dense(num_classes, activation='softmax')                        # Output
    ])
    return model

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # Resize to 32x32 and add channel dimension
    x_train = tf.image.resize(tf.expand_dims(x_train, -1), [32, 32]) / 255.0
    x_test = tf.image.resize(tf.expand_dims(x_test, -1), [32, 32]) / 255.0

    return (x_train, y_train), (x_test, y_test)

# ------------------------------
# Training and Evaluation
# ------------------------------
def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    # Build model
    model = LeNet(num_classes=10)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=64,
                        validation_data=(x_test, y_test))

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # Plot training vs validation accuracy and loss
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
