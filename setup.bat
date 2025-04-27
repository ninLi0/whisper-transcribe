@echo off
:: setup.bat - Script to set up and run the Flask Whisper Transcription App on Windows

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing requirements...
:: Fix requirements with future dates
echo Fixing requirements.txt file dates...
pip install -r requirements.txt 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Some package versions may have issues. Installing without version constraints...
    pip install Flask Werkzeug certifi charset-normalizer decorator ffmpeg filelock fsspec idna imageio
    pip install imageio-ffmpeg Jinja2 llvmlite markdown-it-py MarkupSafe mdurl more-itertools moviepy
    pip install mpmath networkx numba numpy pillow proglog Pygments pytube regex requests rich
    pip install setuptools sympy tiktoken torch tqdm typing_extensions urllib3 wheel
    echo Attempting to install whisper-mps from GitHub...
    pip install git+https://github.com/AtomGradient/whisper-mps.git
)

echo Creating necessary directories...
if not exist uploads mkdir uploads
if not exist recording mkdir recording
if not exist json mkdir json
if not exist transcript mkdir transcript
if not exist templates mkdir templates

echo Creating example audio file...
echo Note: This is just a placeholder. You'll need to upload your own audio/video files. > recording\example.txt

echo Checking for FFmpeg...
ffmpeg -version >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FFmpeg not found. Please install FFmpeg manually:
    echo 1. Download from https://ffmpeg.org/download.html
    echo 2. Add FFmpeg to your system PATH
    echo 3. Restart this script after installation
    
    set /p continue=Do you want to continue without FFmpeg? (y/n): 
    if /i not "%continue%"=="y" (
        echo Setup aborted. Please install FFmpeg and try again.
        exit /b
    )
)

echo Starting Flask application...
echo Open your browser and go to http://localhost:5002
python app.py