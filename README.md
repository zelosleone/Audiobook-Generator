# Audiobook Generator

A high-quality text-to-speech converter that transforms PDF and TXT files into MP4 audio files. Currently using WhisperSpeech technology, but adaptable to future better models.

## Example Output

<audio src="./Audio/Example.mp3" controls>
   Your browser does not support the audio element.
</audio>

## Features

- Supports PDF and TXT input files
- GPU acceleration with CUDA support
- High-quality audio output (44.1kHz, 320kbps AAC)
- Efficient memory management and batch processing
- Multi-threaded CPU processing

## Requirements

- Python 3.x
- NVIDIA GPU with CUDA support (optional)
- Minimum 4GB RAM
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place PDF files in `PDF` directory
2. Place TXT files in `TXT` directory
3. Run:
   ```bash
   python main.py
   ```
4. Find generated audio in `Audio` directory

## Technical Details

- Auto-detects CUDA
- Dynamic batch size optimization
- Large document chunking
- High-quality audio settings
- Memory-efficient processing

## Contributing

Feel free to suggest optimizations or improvements through issues or pull requests. The system is designed to be modular, allowing for easy integration of new TTS models.
