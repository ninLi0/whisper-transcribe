#!/usr/bin/env python3
# whisper_cli.py - Command line tool for transcribing audio/video files using whisper with CUDA/CPU support

import argparse
import json
import os
import re
import subprocess
import sys
import time
import torch
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Check for CUDA availability
def check_cuda():
    """Check if CUDA is available for torch"""
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

# Define folder paths
def create_folders():
    """Create necessary folders for the application"""
    folders = ["recording", "json", "transcript"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    return folders

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

# Main transcription function
def transcribe_file(file_path, model_name="small", sentences_per_paragraph=3, output_format="markdown", use_cuda=True):
    """Transcribe an audio/video file and save the formatted transcript"""
    
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    if not allowed_file(file_path.name):
        logger.error(f"Unsupported file format: {file_path.name}")
        return None
    
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
    
    # Prepare device parameter based on CUDA availability
    device_param = "--device cuda" if use_cuda and check_cuda() else "--device cpu"
    
    # Construct the command
    # First try with whisper (PyTorch/CUDA version)
    try:
        import whisper
        logger.info(f"Using OpenAI Whisper with model: {model_name}")
        logger.info(f"Loading model...")
        
        # Load the model
        model = whisper.load_model(model_name, device="cuda" if use_cuda and check_cuda() else "cpu")
        
        # Start transcription
        logger.info(f"Transcribing file: {file_path}")
        start_time = time.time()
        
        # Transcribe with progress bar
        with tqdm(total=100, desc="Transcribing") as pbar:
            # Fake progress updates as we cannot get real-time progress
            def progress_callback(progress):
                pbar.update(int(progress * 100) - pbar.n)
            
            # Run transcription
            result = model.transcribe(
                str(file_path),
                verbose=True,
                fp16=use_cuda and check_cuda()
            )
            pbar.update(100 - pbar.n)  # Ensure the bar reaches 100%
        
        # Save JSON output
        with open(output_json_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        transcript = result.get("text", "").strip()
        
    except (ImportError, ModuleNotFoundError):
        # Fall back to whisper-mps or other command line tool if import fails
        logger.info("OpenAI Whisper not found. Falling back to whisper-mps...")
        
        # Try whisper-mps first
        command = f"whisper-mps --file-name {file_path} --output-file-name {output_json_file} --model-name {model_name}"
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
            # If whisper-mps fails, try the generic whisper command line tool
            command = f"whisper {file_path} --model {model_name} {device_param} --output_dir {json_folder} --output_format json"
            logger.info(f"whisper-mps failed. Trying generic whisper command: {command}")
            
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
                logger.error(f"Transcription process failed with code {return_code}")
                return None
        
        # Wait until the output JSON file is created
        logger.info("Waiting for output file...")
        timeout = 60
        start_time = time.time()
        while not output_json_file.exists():
            if time.time() - start_time > timeout:
                logger.error("Timed out waiting for the output JSON file.")
                return None
            time.sleep(1)
        
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

def main():
    """Main entry point for the command line interface"""
    # Create parser
    parser = argparse.ArgumentParser(
        description="Whisper Transcription Tool - Command Line Interface"
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
        "--cpu", 
        action="store_true",
        help="Force CPU usage even if CUDA is available"
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
    
    # Create folders
    create_folders()
    
    # Run transcription
    transcribe_file(
        args.file,
        model_name=args.model,
        sentences_per_paragraph=args.sentences,
        output_format=args.format,
        use_cuda=not args.cpu
    )

if __name__ == "__main__":
    main()