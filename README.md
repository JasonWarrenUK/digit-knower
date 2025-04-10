# Digit Knower

> I don't know much, but I know me some digits.

## Current State

### My Understanding

I've had limited time due to facilitating FAC 31 and working on a client project, so I attempted to prioritise grasping the basics of:

- Python
- Torch
- Torchvision

You can find my annotations on the neural network class in my [`train_model.py`](src/train_model.py) file. I think I have a grasp of what each line in the class' `__init__` and `forward` methods is doing, but a lot of it is based on inference.

For instance, I can infer the general purpose of what the `conv*` and `fc*` layers do, but not the mechanics by which that occurs.

### Model

- The model is trained, and saved in the model folder.
- I targeted 85-90% accuracy, but found I was getting 99.24% after 3 epochs (5 phases each)

### UI

- I had reached the point where the user could draw a digit on the canvas, but pressing the predict button failed to acknowledge that a digit had been drawn
- In my subsequent attempts to fix it, I've broken the canvas.
- Sigh.

### Deployment & Containerisation

- Not started

## Overview

Digit Knower is a machine learning project that uses PyTorch to train and deploy a model for digit recognition. The project features a Streamlit-based frontend for interactive testing and PostgreSQL for logging predictions and model performance metrics.

## Components

- **PyTorch Training Pipeline**: Neural network model for digit recognition
- **Streamlit Frontend**: Interactive web interface for testing the model
- **PostgreSQL Database**: Logging system for tracking predictions and model metrics
- **Docker Containerization**: Containerized deployment for consistent environments

## Setup

### Prerequisites

- Python 3.8+
- Docker
- PostgreSQL

### Local Development

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/digit-knower.git
   cd digit-knower
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the development server:

   ```bash
   streamlit run src/app.py
   ```

### Docker Deployment

1. Build the Docker image:

   ```bash
   docker build -t digit-knower .
   ```

2. Run the container:

   ```bash
   docker run -p 8501:8501 digit-knower
   ```
