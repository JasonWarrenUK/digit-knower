import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add the src directory to the Python path so we can import our model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_model import DigitNet

def load_model():
    """Load the pre-trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitNet().to(device)
    model_path = Path('model/mnist_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model 