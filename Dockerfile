# Dockerfile

# Stage 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Copy the dependencies file and install them
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Copy the application code
COPY ./app /app/app
COPY ./scripts /app/scripts
COPY ./data /app/data

# Stage 5: Expose the port the app will run on
EXPOSE 8000

# Stage 6: Define the command to run the application with Gunicorn
# The logging flags ensure that all logs are printed to the console
# so you can see them with 'docker logs' or 'docker-compose logs'.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--log-level=info", "--log-file=-", "app.main:app"]
