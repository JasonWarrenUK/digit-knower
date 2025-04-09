# Implementation Plan

## Overall Blueprint

1. Project Initialization & Setup
   - Create your repository, organize your folders, and add initial documentation.
2. Model Training with PyTorch
   - Write a script to train and evaluate a simple MNIST classifier.
   - Save the model to a file.
3. Interactive Front-End with Streamlit
   - Build a web app with a drawing canvas to input digits.
   - Wire the app to load your model and display predictions and confidence.
   - Add controls for user feedback (correct/incorrect, text input, submit button).
4. Logging Predictions with PostgreSQL
   - Write code that logs each prediction with timestamp, predicted label, and user feedback into a PostgreSQL database.
   - Create the necessary SQL initialization scripts.
5. Containerization using Docker
   - Write a Dockerfile to combine the PyTorch model and Streamlit front-end.
   - Use an official PostgreSQL image for the database.
   - Create a docker-compose.yml file that manages all the containers.
6. Deployment and Documentation
   - Write a README with step-by-step deployment instructions (local, then DigitalOcean).
   - Fully document every command, script, and file so it can be built incrementally.

## Implementation Stages

Below are the series of atomic, iterative prompts you can hand off to a code-generation LLM. Each prompt builds on the previous step with no big jumps in complexity and can be implemented in roughly an hour per prompt.

### Initialize a new Git repository for the MNIST Digit Classifier project

```md
   1. Create a project directory structure with the following layout:
      - /project_root/
         - README.md       # Project overview and instructions.
         - .gitignore      # To exclude virtual environments, __pycache__, etc.
         - /src/           # Source code for the project.
         - /model/         # To store the trained PyTorch model.
         - /docker/        # Docker-related files including Dockerfile and docker-compose.yml.
         - /sql/           # SQL initialization scripts for PostgreSQL.
   2. Create an initial README.md that briefly describes the project and lists the required steps.
   3. Create a basic .gitignore file
      - `__pycache__`
      - `*.pyc`
      - `.env`
      - model files

   Explain each line in the code you write.

   Sample Answer (explain the content):

   - The README.md gives a high-level project overview.
   - The .gitignore ensures build artifacts and environment files are not committed.
```

### Create a minimal README.md and .gitignore for the project

```md
Prompt: Create a minimal README.md and .gitignore for the project

---

README.md should include:

- A project overview.
- A list of major components (PyTorch training, Streamlit frontend, PostgreSQL logging, and Docker containerization).
- Brief instructions on local setup and deployment.

.gitignore should include:

- Python cache directories (__pycache__)
- Compiled Python files (*.pyc)
- The .env file which contains environment variables
- The model directory content (if you don’t want to commit the trained model)

Provide a sample content for both files with detailed comments (explain each line).
```

### Write a PyTorch training script named train_model.py in /src/

```md
Prompt: Write a PyTorch training script named train_model.py in /src/

---

The script must:

1. Load the MNIST dataset.
2. Define a simple neural network that should achieve around 85-90% accuracy.
3. Train the model for a few epochs.
4. Save the trained model to the /model/ folder.

Each section of the code should be commented to explain what it does.

Start by:
- Importing necessary modules (torch, torchvision, torch.nn, torch.optim, etc.).
- Defining a simple neural network architecture.
- Loading the MNIST dataset.
- Writing the training loop (with loss calculation, optimizer step, etc.).
- Saving the model state_dict to a file.

Explain each code line/section for clarity.
```

### Develop a Streamlit front-end script named app.py in /src/

```md
Prompt: Develop a Streamlit front-end script named app.py in /src/

---

1. Loads the pre-trained PyTorch model saved previously.
2. Provides a canvas or drawing area for the user to input a digit.
3. On submission, sends the drawn image to the model for prediction.
4. Displays the predicted digit and confidence score.
5. Provides user feedback controls: a button for 'Correct', a text input for the correct digit if wrong, and a submit button for corrections.

---

Make sure to add detailed comments for every major block and line:

- Explain how the model is loaded.
- Show how to handle image pre-processing to convert the canvas image into tensor format.
- Explain the UI elements and their purpose.
```

### Write a Python module (logger.py) inside /src/

```md
# 1. Connects to a PostgreSQL database using environment variables for configuration.
# 2. Inserts each prediction log into a table with the following fields: timestamp, predicted digit, and user-provided true label.
#
# Include:
# - Code to establish a database connection using psycopg2 (or an alternative library).
# - A function that logs a prediction record (with timestamp, prediction, and feedback).
#
# Ensure every line is commented for clarity:
# - Explain connection handling.
# - Describe the SQL query to insert the data.
```

### Create an SQL script named init_db.sql in the /sql/ folder

```md
# 1. Creates the necessary table to log predictions. The table should include:
#    - An auto-increment primary key.
#    - A timestamp column.
#    - A column for the predicted digit.
#    - A column for the user-provided true label.
#
# Provide comments in the SQL file explaining:
# - What each column represents.
# - Any constraints or defaults you set.
```

### Write a Dockerfile in the /docker/ folder for a combined container

```md
# 1. Sets up a Python environment.
# 2. Installs the necessary packages (PyTorch, Streamlit, psycopg2, etc.).
# 3. Copies the contents of /src/ into the container.
# 4. Defines an ENTRYPOINT or CMD to start the Streamlit app.
#
# Every step in the Dockerfile should include a comment explaining its purpose:
# - E.g., FROM python:3.x -> choosing a base image.
# - RUN commands for installing dependencies.
# - COPY commands for transferring source files.
```

### Create a docker-compose.yml file in the /docker/ folder

```md
# 1. The combined container for the PyTorch and Streamlit application.
# 2. A separate container for PostgreSQL (use the official PostgreSQL image).
#
# The file should:
# - Define services for the web app and database.
# - Set environment variables (you may refer to a .env file) for configuration.
# - Mount volumes if necessary (e.g., for persistent PostgreSQL storage).
#
# Include comments for each section explaining:
# - How the two containers communicate.
# - The purpose of each service and configuration.
```

### Write a detailed README section or a separate DEPLOYMENT.md (inside the /project_root/ folder)

```md
# 1. Instructions to build and run the containers using docker-compose.
# 2. How to test the application locally.
# 3. Steps for deploying the containerized application on a DigitalOcean Droplet (or a similar self-managed server).
#
# This document should:
# - Explain every shell command you instruct (e.g., docker-compose build, docker-compose up).
# - Provide troubleshooting tips for common Docker and deployment issues.
# - Be clear and concise with each step explained so that a developer can follow in no more than a few minutes per step.
```

⸻

## Final Notes

Each step builds on the previous one, ensuring:

- No big jumps: every small change is atomic.
- Complete integration: nothing is left orphaned or unconnected.
- The instructions are clear, contain detailed commentary, and each task is doable in about one hour.
- Prompts are self-explanatory to guide you step by step.

By iterating from initializing the repository, through training, UI integration, logging, containerization, and finally deployment instructions, you have a complete plan that adheres to best practices and ensures incremental progress.

Feel free to adjust the granularity of each prompt if you need even smaller steps; each prompt is already designed to take roughly one hour to complete and review. This series of prompts provides a strong foundation for generating integrated code through a code-generation LLM.