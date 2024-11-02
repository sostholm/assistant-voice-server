# Use the official Python base image
FROM python:3.10-slim

# Install gcc and other necessary build tools
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libportaudio2 \
    portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 9000

# Command to run the application
CMD ["python", "server.py"]