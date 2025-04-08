# Project Architectural Decisions for MNIST Digit Classifier

This document outlines the high-level architectural decisions for our MNIST digit classifier application. The decisions strictly adhere to CRITERIA.md first, then SPEC.md, and finally focus on simplicity.

## 1. Containerization & Deployment

- **Docker Setup:**
  - **Combined Container:** Package both the PyTorch model inference and the Streamlit front-end in a single container.
  - **Separate PostgreSQL Container:** Use a separate container to run the PostgreSQL database.

- **Docker Compose:**
  - Utilize a `docker-compose.yml` file to manage the multi-container setup and orchestrate the services.

- **Environment Variables:**
  - Manage configuration details (e.g., PostgreSQL credentials and connection strings) via a `.env` file for a clear separation of config from code.

## 2. Model Training & Bundling

- **Pre-training:** 
  - Train the PyTorch model locally before containerization.
  - Bundle the pre-trained model file into the container, ensuring that model inference does not delay the container build process.

## 3. Logging

- **Logging Details:**
  - Log each prediction's timestamp, predicted digit, and user-provided true label to PostgreSQL.
  
- **Automated Schema Setup:**
  - Include an initialization SQL script that automatically creates the required table when the PostgreSQL container starts.

## 4. Build & Deployment Automation

- **Automated Scripts:**
  - Provide fully commented, step-by-step shell commands or a simple Makefile in the project repository. This will cover:
    - Building the Docker images.
    - Running the multi-container deployment with Docker Compose.
    - Deploying the application to DigitalOcean.
  
- **Focus on Learning:**
  - Each shell command or Makefile target will be fully commented to explain the purpose of every step, helping you gain a clear understanding of each part of the process.

## 5. Deployment Instructions

- **Minimal & Clear:**
  - Include a clear README with:
    - A project overview.
    - Setup instructions for local testing.
    - Commands for building and containerizing the application.
    - Minimal, yet sufficient, instructions for deployment on DigitalOcean (installation of Docker, Docker Compose, and basic firewall configuration).
  - Note: Since you do not intend to work on this project post-submission, the deployment instructions remain minimal to focus on the initial application process.

## Summary

- **Strict Adherence:** Follow CRITERIA.md and SPEC.md to the letter.
- **Simplified Containerization:** Use a single container for the model and front-end, with a separate container for PostgreSQL.
- **Pre-trained Model:** Train the model locally and bundle it for inference.
- **Automated Setup:** Include initialization scripts for PostgreSQL and fully commented build/deployment instructions via shell scripts or a Makefile.
- **Configuration Management:** Use a `.env` file.
- **DigitalOcean Deployment:** Provide minimal deployment instructions to get the project running quickly.

This architecture ensures that the application is built quickly with an emphasis on simplicity and clarity, while still meeting all specified requirements.