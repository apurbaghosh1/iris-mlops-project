# Dockerfile

# =================================================================
# Dockerfile for the Flask Prediction API
# =================================================================
# This file defines the steps to build a container image for the
# main prediction API. It follows a multi-stage approach for clarity
# and best practices, ensuring a clean and efficient final image.
# =================================================================

# --- Stage 1: Base Image ---
# Start from an official, slim Python base image. 'slim' is a good
# choice as it provides a minimal environment, reducing the final
# image size while still including necessary system libraries.
FROM python:3.9-slim

# --- Stage 2: Set Working Directory ---
# Set the default working directory inside the container. All subsequent
# commands (like COPY, RUN) will be executed from this path.
WORKDIR /app

# --- Stage 3: Install Dependencies ---
# Copy only the requirements file first. Docker caches layers, so this
# step will only be re-run if the requirements.txt file changes. This
# significantly speeds up subsequent builds if only the source code changes.
COPY ./requirements.txt /app/
# Install the Python dependencies. '--no-cache-dir' is a best practice
# that reduces the size of the Docker layer by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy Application Source Code ---
# Copy the application logic, training scripts, and data into the container.
# This is done after installing dependencies so that code changes don't
# invalidate the dependency layer cache.
COPY ./app /app/app
COPY ./scripts /app/scripts
COPY ./data /app/data

# --- Stage 5: Expose Port ---
# Inform Docker that the container listens on port 8000 at runtime.
# This is documentation; the actual port mapping is done in docker-compose.yml
# or with the 'docker run -p' flag.
EXPOSE 8000

# --- Stage 6: Define Runtime Command ---
# Specify the command to run when the container starts.
# We use Gunicorn, a production-grade WSGI server, instead of Flask's
# built-in development server.
# --bind 0.0.0.0:8000: Binds the server to all network interfaces inside the container.
# --log-level=info: Sets the logging verbosity.
# --log-file=-: Directs logs to stdout/stderr, which is the standard for
#   containerized applications, allowing Docker to manage the logs.
# app.main:app: Tells Gunicorn to find the 'app' object inside the 'app/main.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--log-level=info", "--log-file=-", "app.main:app"]
