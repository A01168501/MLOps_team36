import os

def create_directories(directories):
    """Create directories if they do not exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

if __name__ == '__main__':
    # List of directories you want to ensure exist
    directories_to_create = [
        'data/raw',
        'data/processed',
        'data/final',
        'data/external',
        'results/experiments',
        'results/predictions',
        'results/logs'
    ]

    # Create directories
    create_directories(directories_to_create)