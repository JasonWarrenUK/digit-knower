# MLX Digit Classifier Application - Specification

This document outlines a clear, minimal, step-by-step specification for developing, containerizing, and deploying an MNIST digit recognition application.

---

## Project Overview

Develop an end-to-end digit recognition application using PyTorch for model training, Streamlit for frontend interaction, PostgreSQL for logging predictions, and Docker for containerization. Deployment will be via DigitalOcean, using a server public IP address.

---

## Specifications

### 1. Model Training (PyTorch)

- **Dataset:** MNIST handwritten digits.
- **Framework:** PyTorch.
- **Target Accuracy:**
  - Approximately 85–90% accuracy is sufficient.
  - Prioritize simplicity and functionality over accuracy optimization.

### 2. Interactive Front-End (Streamlit)

- **Drawing Input:**
  - Freeform canvas allowing users to draw digits.
- **Output:**
  - Prediction (digit 0–9).
  - Confidence score/probability.
- **User Feedback:**
  - Three mandatory feedback elements arranged left to right:
    1. **Correct Button:** Users confirm a correct prediction.
    2. **Text Input:** Users enter the correct digit if incorrect.
    3. **Submit Correction Button:** Users submit their correction.
- **Behavior After Submission:**
  - Immediately clear the drawing canvas.
  - Reset all prediction outputs and feedback inputs.
- **Historical Feedback:**
  - No historical prediction or accuracy statistics displayed.

### 3. Logging (PostgreSQL)

- **Data to Log:**
  - Timestamp of prediction.
  - Predicted digit.
  - User-provided true label.
- **Additional Logging:**
  - None; strictly adhere to specified fields.

### 4. Containerization (Docker)

- **Container Structure:**
  - Combined container for PyTorch model inference and Streamlit frontend application.
  - Separate container for PostgreSQL database.
- **Management:**
  - Use `docker-compose.yml` for clear, simple multi-container setup.

### 5. Deployment (DigitalOcean)

- **Hosting Provider:**
  - DigitalOcean selected for beginner-friendly setup, clear UX, and ease of use.
- **Access:**
  - Deploy application accessible directly via the server’s public IP.
- **Domain:**
  - No custom domain required; use provided public IP.

### 6. GitHub Repository

- **Documentation (README):**
  - Clear and detailed README including:
    - Project overview.
    - Setup and local testing instructions.
    - Docker setup and deployment instructions.
    - URL to deployed application.
- **Additional Assets:**
  - Scripts and commands required to replicate environment setup and deployment explicitly provided.

---

## Development Priorities (KISS Principles)

- Emphasis on simplicity, clarity, and ease of debugging.
- Minimal viable accuracy and feature set prioritized over complexity.
- Ensure easy troubleshooting due to your inexperience with Docker and DevOps tasks.

---

This specification is structured to ensure clarity, ease of implementation, and straightforward debugging, perfectly suited to complete within the available time constraints (one week, intermittent availability).