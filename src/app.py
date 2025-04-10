import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os
import base64
import io

# Import our modules
from src.streamlit_canvas import canvas_component, get_image_from_canvas
from src.model_loader import load_model

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
    image = image.resize((28, 28), Image.LANCZOS)

    
    # Convert to tensor and normalize
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension

def predict_digit(model, image_tensor):
    """Make a prediction using the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_digit = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_digit].item()
    return predicted_digit, confidence

# Set up the Streamlit interface
st.title('Digit Recognition App')
st.write('Draw a digit (0-9) in the box below')

# Load the model
model = load_model()

# Display the canvas
canvas_component()

# Add prediction and feedback controls
col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Predict'):
        # Get the image data from the canvas
        image = get_image_from_canvas()
        if image:
            # Preprocess the image
            tensor = preprocess_image(image)
            # Get prediction
            predicted_digit, confidence = predict_digit(model, tensor)
            # Store results in session state
            st.session_state.predicted_digit = predicted_digit
            st.session_state.confidence = confidence
            st.session_state.show_results = True
        else:
            st.warning("Please draw a digit first.")

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