import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the Python path so we can import our model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_model import DigitNet

# Initialize session state for storing the model
if 'model' not in st.session_state:
    # Load the pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitNet().to(device)
    model_path = Path('model/mnist_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    st.session_state.model = model

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

def preprocess_image(image):
    """Convert the drawn image to the format expected by the model."""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28 (MNIST size)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension

def predict_digit(image_tensor):
    """Make a prediction using the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = st.session_state.model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_digit = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_digit].item()
    return predicted_digit, confidence

# Set up the Streamlit interface
st.title('Digit Recognition App')
st.write('Draw a digit (0-9) in the box below')

# Create a canvas for drawing
canvas_result = st.canvas(
    fill_color='black',
    stroke_width=20,
    stroke_color='white',
    drawing_mode='freedraw',
    update_streamlit=True,
    height=280,
    width=280,
    key='canvas'
)

# Add prediction and feedback controls
col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Predict'):
        if canvas_result.image_data is not None:
            # Convert the canvas image to PIL Image
            image = Image.fromarray(canvas_result.image_data)
            # Preprocess the image
            tensor = preprocess_image(image)
            # Get prediction
            predicted_digit, confidence = predict_digit(tensor)
            # Store results in session state
            st.session_state.predicted_digit = predicted_digit
            st.session_state.confidence = confidence
            st.session_state.show_results = True

with col2:
    if 'show_results' in st.session_state and st.session_state.show_results:
        st.write(f'Predicted Digit: {st.session_state.predicted_digit}')
        st.write(f'Confidence: {st.session_state.confidence:.2%}')

with col3:
    if 'show_results' in st.session_state and st.session_state.show_results:
        correct_digit = st.text_input('Correct Digit (if wrong)', key='correct_digit')
        if st.button('Submit Feedback'):
            # Here you would typically log the prediction and feedback to your database
            st.success('Feedback submitted!')
            # Clear the canvas and results
            st.session_state.show_results = False
            st.experimental_rerun() 