# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies if required for compiling certain packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements first (to leverage Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code (including static/ folder and ML model weights/pickles)
COPY . .

# Expose the port FastAPI runs on
EXPOSE 80

# Run Uvicorn production server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]