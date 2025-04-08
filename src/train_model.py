import torch # note: torch is the baseline tensor library
import torch.nn as nn # note: neural network module 
import torch.optim as optim # note: optimisation module
import torchvision # note: vision
import torchvision.transforms as transforms # note: transforming images to tensors
from pathlib import Path # note: filepaths as objects

torch.manual_seed(23051985)  # note: don't think the seed matters as long as it's consistent, so... let's use me birthday

class DigitNet(nn.Module):
  def __init__(self):
    super(DigitNet, self).__init__()
    # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
    # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    # Max pooling layer
    self.pool = nn.MaxPool2d(2)
    # Dropout layer to prevent overfitting
    self.dropout = nn.Dropout(0.25)
    # Fully connected layers
    self.fc1 = nn.Linear(64 * 5 * 5, 128)
    self.fc2 = nn.Linear(128, 10)  # 10 output classes for digits 0-9

  def forward(self, x):
    # First conv block: conv -> relu -> pool
    x = self.pool(torch.relu(self.conv1(x)))
    # Second conv block: conv -> relu -> pool
    x = self.pool(torch.relu(self.conv2(x)))
    # Flatten the tensor for fully connected layers
    x = x.view(-1, 64 * 5 * 5)
    # Dropout and fully connected layers
    x = self.dropout(torch.relu(self.fc1(x)))
    x = self.fc2(x)
    return x

def main():
  # Set device (GPU if available, else CPU)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")

  # Define data transformations
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
  ])

  # Load MNIST dataset
  train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
  )

  # Create data loader
  train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
  )

  # Initialize model, loss function, and optimizer
  model = DigitNet().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Training loop
  num_epochs = 5
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
      # Move data to device
      images, labels = images.to(device), labels.to(device)

      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward pass and optimize
      loss.backward()
      optimizer.step()

      # Statistics
      running_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # Print progress every 100 batches
      if (i + 1) % 100 == 0:
        print(
          f'Epoch [{epoch+1}/{num_epochs}], '
          f'Step [{i+1}/{len(train_loader)}], '
          f'Loss: {running_loss/100:.4f}, '
          f'Accuracy: {100 * correct / total:.2f}%'
        )
        running_loss = 0.0

  # Create model directory if it doesn't exist
  model_dir = Path('model')
  model_dir.mkdir(exist_ok=True)

  # Save the model
  model_path = model_dir / 'mnist_model.pth'
  torch.save(model.state_dict(), model_path)
  print(f'\nModel saved to {model_path}')

if __name__ == '__main__':
  main() 