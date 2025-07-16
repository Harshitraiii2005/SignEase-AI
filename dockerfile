# Base Python image
FROM python:3.10

# Set working directory inside container
WORKDIR /app/App

# Install system dependencies (required by OpenCV, TFLite etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python packages
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt

# Copy entire project directory
COPY . .

# Set Python module path to include the App directory
ENV PYTHONPATH=/app/App

# Expose Flask default port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
