#!/usr/bin/env python3
# whisper_windows.py - Windows-friendly command line tool for transcribing audio/video files

import argparse
import json
import os
import re
import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define folder paths - using Path for cross-platform compatibility
def create_folders():
    """Create necessary folders for the application"""
    folders = ["recording", "json", "transcript"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    return True

# Check if the file has an allowed extension
def allowed_file(filename):
    """Check if the file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mp3', 'wav', 'flac', 'm4a'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Format transcript into paragraphs
def format_transcript(text, sentences_per_paragraph=3, base_name="", output_format="markdown"):
    """Format the transcript into paragraphs"""
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
        for paragraph in paragraphs:
            formatted_text += f"{paragraph}\n\n"
        return formatted_text
    else:
        return "\n\n".join(paragraphs)

# Check for torch and CUDA availability
def check_torch():
    """Check if PyTorch is available and configured with CUDA"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"CUDA is available (version {cuda_version})")
            logger.info(f"Found {device_count} CUDA device(s): {device_name}")
        else:
            logger.warning("CUDA is not available, falling back to CPU")
        
        return cuda_available
    except ImportError:
        logger.warning("PyTorch not installed")
        return False

# Check if FFmpeg is installed
def check_ffmpeg():
    """Check if FFmpeg is installed and in PATH"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info("FFmpeg is installed")
            return True
        else:
            logger.warning("FFmpeg check failed")
            return False
    except FileNotFoundError:
        logger.warning("FFmpeg not found in PATH")
        return False

# Try different transcription methods
def transcribe_file(file_path, model_name="small", sentences_per_paragraph=3, output_format="markdown"):
    """Try different transcription methods and return first success"""
    
    # Use proper Path objects for Windows compatibility
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    if not allowed_file(file_path.name):
        logger.error(f"Unsupported file format: {file_path.name}")
        return None
    
    # Ensure folders exist
    create_folders()
    
    # Extract base name from file (without extension)
    base_name = file_path.stem
    
    # Set file paths for JSON and transcript outputs
    json_folder = Path("json")
    transcript_folder = Path("transcript")
    
    output_json_file = json_folder / f"{base_name}_output.json"
    
    # Determine file extension based on format
    file_extension = "md" if output_format == "markdown" else "txt"
    transcript_file = transcript_folder / f"{base_name}_transcript.{file_extension}"
    
    # Ensure no outdated JSON file exists
    if output_json_file.exists():
        logger.info("Deleting old JSON to prevent stale data...")
        output_json_file.unlink()
    
    # Record start time
    start_time = time.time()
    
    # Try different transcription methods in order of preference
    methods = [
        try_openai_whisper,
        try_whisper_mps,
        try_whisper_cli
    ]
    
    for method in methods:
        logger.info(f"Attempting transcription with {method.__name__}...")
        result = method(file_path, output_json_file, model_name)
        if result:
            # If method was successful, we have output JSON
            break
    else:
        # If we get here, all methods failed
        logger.error("All transcription methods failed")
        return None
    
    # Read and parse the JSON file
    try:
        logger.info("Reading transcription output...")
        with open(output_json_file, "r", encoding="utf-8") as file:
            output_json = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error reading output JSON: {str(e)}")
        return None
    
    # Extract transcript text
    transcript = output_json.get("text", "").strip()
    if not transcript:
        logger.error("Transcription failed or returned empty text.")
        return None
    
    # Format the transcript into paragraphs
    logger.info("Formatting transcript...")
    formatted_transcript = format_transcript(
        transcript, 
        sentences_per_paragraph=sentences_per_paragraph,
        base_name=base_name,
        output_format=output_format
    )
    
    # Save the formatted transcript
    logger.info(f"Saving transcript to {transcript_file}...")
    with open(transcript_file, "w", encoding="utf-8") as file:
        file.write(formatted_transcript)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds!")
    logger.info(f"Transcript saved to: {transcript_file}")
    
    return transcript_file

# Method 1: Try using OpenAI's whisper Python package directly
def try_openai_whisper(file_path, output_json_file, model_name):
    """Attempt to transcribe using OpenAI's whisper package"""
    try:
        import whisper
        logger.info(f"Using OpenAI Whisper with model: {model_name}")
        logger.info(f"Loading model...")
        
        use_cuda = check_torch()
        device = "cuda" if use_cuda else "cpu"
        
        # Load the model
        model = whisper.load_model(model_name, device=device)
        
        # Start transcription
        logger.info(f"Transcribing file: {file_path}")
        
        # Run transcription
        result = model.transcribe(
            str(file_path),
            verbose=True,
            fp16=use_cuda
        )
        
        # Save JSON output
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"OpenAI Whisper transcription completed successfully")
        return True
        
    except ImportError:
        logger.warning("OpenAI Whisper package not installed")
        return False
    except Exception as e:
        logger.error(f"Error with OpenAI Whisper: {str(e)}")
        return False

# Method 2: Try using whisper-mps
def try_whisper_mps(file_path, output_json_file, model_name):
    """Attempt to transcribe using whisper-mps"""
    try:
        # Windows-friendly command construction
        command = f"whisper-mps --file-name \"{file_path}\" --output-file-name \"{output_json_file}\" --model-name {model_name}"
        logger.info(f"Running command: {command}")
        
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
        
        # Display output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"whisper-mps process failed with code {return_code}")
            return False
        
        # Wait until the output JSON file is created
        logger.info("Waiting for output file...")
        timeout = 60
        start_time = time.time()
        while not os.path.exists(output_json_file):
            if time.time() - start_time > timeout:
                logger.error("Timed out waiting for the output JSON file.")
                return False
            time.sleep(1)
        
        logger.info(f"whisper-mps transcription completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error with whisper-mps: {str(e)}")
        return False

# Method 3: Try using the generic whisper command line tool
def try_whisper_cli(file_path, output_json_file, model_name):
    """Attempt to transcribe using the whisper command line tool"""
    try:
        use_cuda = check_torch()
        device_param = "--device cuda" if use_cuda else "--device cpu"
        
        json_folder = os.path.dirname(output_json_file)
        
        # Windows-friendly command construction
        command = f"whisper \"{file_path}\" --model {model_name} {device_param} --output_dir \"{json_folder}\" --output_format json"
        logger.info(f"Running command: {command}")
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Display output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"whisper CLI process failed with code {return_code}")
            return False
        
        # CLI might output with a different filename pattern
        # Try to find the JSON file created
        base_name = os.path.basename(file_path).rsplit('.', 1)[0]
        possible_files = [
            os.path.join(json_folder, f"{base_name}.json"),
            output_json_file
        ]
        
        for potential_file in possible_files:
            if os.path.exists(potential_file) and potential_file != output_json_file:
                # If found but not with our expected name, copy it
                with open(potential_file, 'r') as src, open(output_json_file, 'w') as dst:
                    dst.write(src.read())
                break
        
        # Wait until the output JSON file is created
        logger.info("Waiting for output file...")
        timeout = 60
        start_time = time.time()
        while not os.path.exists(output_json_file):
            if time.time() - start_time > timeout:
                logger.error("Timed out waiting for the output JSON file.")
                return False
            time.sleep(1)
        
        logger.info(f"whisper CLI transcription completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error with whisper CLI: {str(e)}")
        return False

def main():
    """Main entry point for the command line interface"""
    # Create parser
    parser = argparse.ArgumentParser(
        description="Windows-friendly Whisper Transcription Tool"
    )
    
    # Add arguments
    parser.add_argument(
        "file", 
        help="Path to the audio/video file to transcribe"
    )
    parser.add_argument(
        "--model", 
        choices=["tiny", "base", "small", "medium", "large"], 
        default="small",
        help="Whisper model to use (default: small)"
    )
    parser.add_argument(
        "--sentences", 
        type=int, 
        default=3,
        help="Number of sentences per paragraph (default: 3)"
    )
    parser.add_argument(
        "--format", 
        choices=["markdown", "text"], 
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check for prerequisites
    logger.info("Checking prerequisites...")
    check_torch()
    check_ffmpeg()
    
    # Run transcription
    transcribe_file(
        args.file,
        model_name=args.model,
        sentences_per_paragraph=args.sentences,
        output_format=args.format
    )

if __name__ == "__main__":
    main()