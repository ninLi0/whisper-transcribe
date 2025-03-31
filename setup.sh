#!/bin/bash
# setup.sh - Script to set up and run the Flask Whisper Transcription App

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads recording json transcript templates

# Create example to show how it works
echo "Creating example audio file..."
echo "Note: This is just a placeholder. You'll need to upload your own audio/video files."
touch recording/example.txt

# Start the Flask app
echo "Starting Flask application..."
echo "Open your browser and go to http://localhost:5000"
python app.py