# Whisper Transcription Web App

A Flask-based web interface for transcribing audio and video files using whisper-mps and generating Markdown formatted transcripts.

## Features

- Web-based interface for easy file uploads
- Uses whisper-mps for high-quality transcriptions
- Generates formatted Markdown transcripts
- Supports multiple whisper models (tiny, base, small, medium, large)
- Adjustable paragraph formatting
- View, download, and manage transcripts through the web interface

## Directory Structure

```
whisper-web-app/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   ├── base.html          # Base template with common elements
│   ├── index.html         # Home page with upload form and transcript list
│   └── view.html          # Transcript viewing page
├── uploads/               # Temporary storage for uploaded files
├── recording/             # Storage for files to be processed
├── json/                  # Output JSON files from whisper-mps
└── transcript/            # Generated transcript files (Markdown)
```

## Installation

1. Clone this repository or download the files

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5002
   ```

3. Upload an audio or video file through the web interface

4. Select the desired whisper model and number of sentences per paragraph

5. Click "Upload & Transcribe" to start the transcription process

6. Once complete, the transcript will appear in the list below the upload form

7. You can view, download, or delete transcripts from the web interface

## Supported File Formats

- Audio: MP3, WAV, FLAC, M4A
- Video: MP4, AVI, MOV

## Whisper Models

- tiny: Fastest, less accurate
- base: Fast, reasonable accuracy
- small: Good balance between speed and accuracy (recommended)
- medium: More accurate, slower
- large: Most accurate, slowest

## Notes

- The transcription process runs on your local machine and can be resource-intensive, especially with larger models
- Processing time depends on the length of the audio/video file and the chosen model
- All transcripts are saved in Markdown format for easy formatting and readability
- The application stores all files locally on your machine

## Folder Organization

- **uploads/**: Temporary storage for uploaded files
- **recording/**: Files ready for processing by whisper-mps
- **json/**: JSON output files from the whisper-mps process
- **transcript/**: Final Markdown transcript files

## Customize

You can modify the templates in the `templates/` directory to change the appearance of the web interface. The Flask application code in `app.py` can be adjusted to add new features or change behavior.

## Dependencies

This application relies on:
- Flask for the web interface
- whisper-mps for the transcription functionality
- Bootstrap for styling
- marked.js for Markdown rendering