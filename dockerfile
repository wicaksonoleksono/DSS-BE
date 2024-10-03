# Use the official lightweight Python image.
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container.
WORKDIR /app

# Copy and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Add src/ to PYTHONPATH.
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Expose the port your app runs on.
EXPOSE 8080

# Command to run the application.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "index:app"]
