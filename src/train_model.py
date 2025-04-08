import torch
# note: torch is the baseline tensor library
import torch.nn as nn
# note: neural network module 
import torch.optim as optim
# note: optimisation module
import torchvision
# note: vision
import torchvision.transforms as transforms
# note: transforming images to tensors
from pathlib import Path
# note: filepaths as objects

torch.manual_seed(23051985)
# note: don't think the seed matters as long as it's consistent, so... let's use me birthday

# =1 Neural Network Class
class DigitNet(nn.Module):
  # =2 Constructor Method
  def __init__(self):
    # ??: how is this called? is it automatic when .init called? Implies different use of `.` chain to JS
    super(DigitNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
    # -2 `self` = this class, `conv1` = the first convolutional layer
      # -3 so actually the `.` operator is doing the same as JS
      # -3 am I defining a prop or a method?
    # -2 `nn` is the neural network model I imported
      # -3 so this is shorthand for `torch.nn`
    # -2 Conv2d is a method(?) of `torch.nn`
      # -3 so this is shorthand for `torch.nn.Conv2d`
      # -3 I can infer that Conv2d is... convolution 2D image?
      # -3 takes 3 args
        # ?? input is a 1 channel image (?)
        # ?? output is a 32 channel image (?)
        # ?? 3 means a 3x3 filter (?)
        # ?? WAIT or is it a 1 channel image and then 32 filtered images. different resolutions to trace edges?
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    # -2 So presumably this does the same, converting 32 channels to 64
      # ?? composite together the 32 maps and convert to 64 maps?
      # ?? separately pass 32 maps and generate 64 maps?
      # ?? pass in a 32 channel image and generate a 64 channel image?
    self.pool = nn.MaxPool2d(2)
    # -2 So this looks like methods that execute sequentially, so this is something that gets done to the 64
    # -2 MaxPool2d
      # -3 so we're applying a ceiling
      # ?? to a "pool" (?)
      # -3 to a 2d image
      # ?? so this is about making the expanded information from the convolutional layers manageable?
      # !! fc1 is taking 128 (64 * 2) as an arg
      # ?? so is this pooling together up to 2 of `self.conv2`?
    self.dropout = nn.Dropout(0.25)
    # -2 this looks like a probability
    # -2 we're doing something 25% of the time
    # ?? maybe... this is about introducing friction? cancelling 25% of methods to approximate the chaos of reality?
    self.fc1 = nn.Linear(64 * 5 * 5, 128)
    # -2 welp no idea what this does, BUT...
      # -3 conv1 & conv2 increased (1, 32, 64)
      # -3 fc1 & fc2 decreased (128, 10)
    self.fc2 = nn.Linear(128, 10)
    # -2 so it looks like these two steps were about condensing the expanded detail into an output

  # =2 this calls a method, I think x is initially an image
  def forward(self, x):
    # -2 OH HELLO I think I misunderstood.
    # -2 The `__init__` methods don't flow in order.
    # -2 They're passed by these `forward` methods.
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    # !! 2 pools! self.pool = nn.MaxPool2d(2)!!
    # ?? x is an image?
    # -2 relu defines a tensor.
    x = x.view(-1, 64 * 5 * 5)
    # -2 ok so... 64 * 5 is 320, * 5 is 1600
    x = self.dropout(torch.relu(self.fc1(x)))
    x = self.fc2(x)
    # ?? this doesn't get converted to a tensor?
    return x

# note: okay, so far I'm inferring that there's this constant push-pull between expanding detail then constraining output.

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