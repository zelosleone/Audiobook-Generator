# Audiobook Generator

A high-quality text-to-speech converter that transforms PDF and TXT files into MP4 audio files. Currently using WhisperSpeech technology, but adaptable to future better models.

## Example Output

https://github.com/user-attachments/assets/637660f8-7cc8-492f-b4f4-764cbbb3d9bd

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

### Performance Optimizations

- CUDA-aware processing with automatic GPU detection
- Dynamic batch sizing based on available VRAM/RAM
- Multi-threaded CPU processing for non-GPU operations
- Memory-efficient chunking for large documents

### Audio Processing

- 44.1kHz sampling rate
- 320kbps AAC encoding
- Stereo output
- Zero-quality loss audio settings

### System Architecture

- Modular pipeline design for easy model swapping
- Buffered I/O operations (1MB buffer)
- Automatic memory management with CUDA cache clearing
- Fault-tolerant processing with error handling

### Resource Management

- Dynamic worker allocation based on system specs
- Configurable chunk sizes (default: 2000 tokens)
- Adaptive batch processing
- Progressive audio concatenation

## Contributing

Feel free to suggest optimizations or improvements through issues or pull requests. The system is designed to be modular, allowing for easy integration of new TTS models.
