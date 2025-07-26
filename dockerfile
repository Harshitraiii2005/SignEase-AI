FROM python:3.10

# Set root working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy entire project
COPY . .

# Set Python module path to include App
ENV PYTHONPATH=/app/App

# Upgrade pip & install dependencies
RUN python -m pip install --upgrade pip && \
    pip install --default-timeout=300 --retries=10 --no-cache-dir -r requirements.txt

# Set working directory to where app.py is
WORKDIR /app/App

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
