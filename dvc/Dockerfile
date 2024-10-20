# Use base python image
FROM python:3.10-slim


# Install Git
RUN apt-get update && apt-get install -y git && apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt .  

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project to the container
COPY . .    

# Set the default command to run when starting the container
CMD ["dvc", "repro"]