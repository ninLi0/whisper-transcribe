# app.py - Flask web interface for transcribing and formatting text using whisper-mps with progress tracking

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, Response
import subprocess
import json
import time
import os
import re
import logging
import shutil
import datetime
import threading
import queue
import uuid
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
app.secret_key = "whisper_transcription_app"  # Secret key for flash messages

# Define folder paths
UPLOAD_FOLDER = "uploads"
RECORDING_FOLDER = "recording"
JSON_FOLDER = "json"
TRANSCRIPT_FOLDER = "transcript"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mp3', 'wav', 'flac', 'm4a'}

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, RECORDING_FOLDER, JSON_FOLDER, TRANSCRIPT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Store active transcription jobs and their progress queues
active_jobs = {}

# Define the timestamp_to_date filter
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """Convert timestamp to readable date format"""
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_worker(file_name, model_name, sentences_per_paragraph, output_format, job_id):
    """
    Background worker function that handles the transcription process
    and captures the progress output.
    """
    progress_queue = active_jobs[job_id]['queue']
    
    # Construct the full path for the input video file
    file_path = os.path.join(RECORDING_FOLDER, file_name)
    
    # Check if the file exists in the recording folder
    if not os.path.exists(file_path):
        logging.error(f"File '{file_name}' not found in the '{RECORDING_FOLDER}' folder.")
        progress_queue.put({"status": "error", "message": f"File not found: {file_name}"})
        return None
    
    # Extract base name from file (without extension)
    base_name = os.path.splitext(file_name)[0]
    
    # Set file paths for JSON and transcript outputs
    output_json_file = os.path.join(JSON_FOLDER, f"{base_name}_output.json")
    
    # Determine file extension based on format
    file_extension = "md" if output_format == "markdown" else "txt"
    transcript_file = os.path.join(TRANSCRIPT_FOLDER, f"{base_name}_transcript.{file_extension}")
    
    # Ensure no outdated JSON file exists
    if os.path.exists(output_json_file):
        logging.info("Deleting old JSON to prevent stale data...")
        os.remove(output_json_file)
    
    # Progress update
    progress_queue.put({"status": "preparing", "message": "Preparing to transcribe..."})
    
    # Construct the command
    command = f"whisper-mps --file-name {file_path} --output-file-name {output_json_file} --model-name {model_name}"
    
    try:
        # Execute transcription command with output capture
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Send process output to the client
        for line in iter(process.stdout.readline, ''):
            if line:
                progress_queue.put({"status": "processing", "message": line.strip()})
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            progress_queue.put({"status": "error", "message": f"Transcription process failed with code {return_code}"})
            return None
        
        # Wait until the output JSON file is created (max wait time: 60 sec)
        progress_queue.put({"status": "finalizing", "message": "Waiting for output file..."})
        timeout = 60
        start_time = time.time()
        while not os.path.exists(output_json_file):
            if time.time() - start_time > timeout:
                progress_queue.put({"status": "error", "message": "Timed out waiting for the output JSON file."})
                return None
            time.sleep(1)
        
        # Read and parse the JSON file
        try:
            progress_queue.put({"status": "formatting", "message": "Reading transcription output..."})
            with open(output_json_file, "r", encoding="utf-8") as file:
                output_json = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            progress_queue.put({"status": "error", "message": f"Error reading output JSON: {str(e)}"})
            return None
        
        # Extract transcript text
        transcript = output_json.get("text", "").strip()
        if not transcript:
            progress_queue.put({"status": "error", "message": "Transcription failed or returned empty text."})
            return None
        
        # Format the transcript into paragraphs
        progress_queue.put({"status": "formatting", "message": "Formatting transcript..."})
        
        def format_transcript(text, sentences_per_paragraph=3):
            sentences = re.split(r"(?<=[.!?])\s+", text)
            paragraphs = [
                " ".join(sentences[i: i + sentences_per_paragraph])
                for i in range(0, len(sentences), sentences_per_paragraph)
            ]
            
            # For markdown, add formatting
            if output_format == "markdown":
                # Add title
                formatted_text = f"# Transcript: {base_name}\n\n"
                # Add each paragraph
                for idx, paragraph in enumerate(paragraphs):
                    formatted_text += f"{paragraph}\n\n"
                return formatted_text
            else:
                return "\n\n".join(paragraphs)
        
        formatted_transcript = format_transcript(transcript, sentences_per_paragraph)
        
        # Save the formatted transcript
        progress_queue.put({"status": "saving", "message": "Saving transcript..."})
        with open(transcript_file, "w", encoding="utf-8") as file:
            file.write(formatted_transcript)
        
        progress_queue.put({"status": "complete", "message": f"Transcription complete!", "result": transcript_file})
        logging.info(f"Formatted transcription saved to {transcript_file}")
        return transcript_file
    
    except Exception as e:
        progress_queue.put({"status": "error", "message": f"Error during transcription: {str(e)}"})
        logging.error(f"Error during transcription: {e}")
        return None

@app.route('/')
def index():
    """Render the main page"""
    # Get list of available transcripts
    transcripts = []
    for file in os.listdir(TRANSCRIPT_FOLDER):
        if file.endswith('.md') or file.endswith('.txt'):
            transcripts.append({
                'name': file,
                'path': os.path.join(TRANSCRIPT_FOLDER, file),
                'date_modified': os.path.getmtime(os.path.join(TRANSCRIPT_FOLDER, file))
            })
    
    # Sort transcripts by date modified (newest first)
    transcripts.sort(key=lambda x: x['date_modified'], reverse=True)
    
    return render_template('index.html', transcripts=transcripts)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start transcription"""
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save to uploads folder first
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        
        # Copy to recording folder for processing
        recording_path = os.path.join(RECORDING_FOLDER, filename)
        shutil.copy2(upload_path, recording_path)
        
        # Get form parameters
        model = request.form.get('model', 'small')
        sentences = int(request.form.get('sentences', 3))
        
        # Create a unique job ID
        job_id = str(uuid.uuid4())
        
        # Create a queue for this job and store in active_jobs
        active_jobs[job_id] = {
            'queue': queue.Queue(),
            'filename': filename
        }
        
        # Start the transcription process in a background thread
        threading.Thread(
            target=transcribe_worker,
            args=(filename, model, sentences, "markdown", job_id),
            daemon=True
        ).start()
        
        # Redirect to the progress page
        return redirect(url_for('show_progress', job_id=job_id))
    
    flash('File type not allowed', 'error')
    return redirect(url_for('index'))

@app.route('/progress/<job_id>')
def show_progress(job_id):
    """Show the progress page for a specific job"""
    if job_id not in active_jobs:
        flash('Invalid job ID', 'error')
        return redirect(url_for('index'))
    
    filename = active_jobs[job_id]['filename']
    return render_template('progress.html', job_id=job_id, filename=filename)

@app.route('/progress-stream/<job_id>')
def progress_stream(job_id):
    """Stream progress updates for a specific job"""
    def generate():
        if job_id not in active_jobs:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Invalid job ID'})}\n\n"
            return
        
        # Get the queue for this job
        progress_queue = active_jobs[job_id]['queue']
        
        # We'll wait for updates for up to 30 minutes (1800 seconds)
        end_time = time.time() + 1800000
        
        while time.time() < end_time:
            try:
                # Try to get a message from the queue
                message = progress_queue.get(timeout=1.0)
                yield f"data: {json.dumps(message)}\n\n"
                
                # If the job is complete or encountered an error, we're done
                if message['status'] in ['complete', 'error']:
                    # Clean up the job after a short delay
                    def cleanup_job():
                        time.sleep(10)  # Wait 10 seconds before cleanup
                        if job_id in active_jobs:
                            del active_jobs[job_id]
                    
                    threading.Thread(target=cleanup_job, daemon=True).start()
                    break
            except queue.Empty:
                # Send a heartbeat to keep the connection alive
                yield f"data: {json.dumps({'status': 'heartbeat'})}\n\n"
        
        # Final cleanup if we timed out
        if time.time() >= end_time and job_id in active_jobs:
            del active_jobs[job_id]
            yield f"data: {json.dumps({'status': 'error', 'message': 'Transcription timed out'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/transcripts/<filename>')
def download_transcript(filename):
    """Download a transcript file"""
    return send_from_directory(TRANSCRIPT_FOLDER, filename)

@app.route('/view/<filename>')
def view_transcript(filename):
    """View a transcript file in the browser"""
    try:
        with open(os.path.join(TRANSCRIPT_FOLDER, filename), 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template('view.html', filename=filename, content=content)
    except FileNotFoundError:
        flash('Transcript not found', 'error')
        return redirect(url_for('index'))

@app.route('/delete/<filename>', methods=['POST'])
def delete_transcript(filename):
    """Delete a transcript file"""
    try:
        os.remove(os.path.join(TRANSCRIPT_FOLDER, filename))
        flash(f'Transcript {filename} deleted', 'success')
    except FileNotFoundError:
        flash('Transcript not found', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)