# Use base python image
FROM python:3.10.15

# Establish directory
WORKDIR /app

# Copy the project requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port where MLFlow serves the UI.
EXPOSE 5000

# Execute mlflow
CMD ["mlflow", "ui", "--host", "0.0.0.0"]