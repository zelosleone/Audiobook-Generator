import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import PyPDF2
import soundfile as sf
from whisperspeech.pipeline import Pipeline as WhisperPipeline
import psutil
from moviepy import AudioArrayClip

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("\n=== GPU Diagnostics ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
else:
    print("Possible reasons for CUDA not available:")
    print("1. PyTorch not installed with CUDA support")
    print("2. NVIDIA drivers not properly installed, install Nvidia CUDA from developers portal.")
    print("3. GPU not CUDA-compatible")
    print("\nTry running: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
print("=====================\n")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

CHUNK_SIZE = 1000
SAMPLE_RATE = 44100  # Increased to CD quality

def get_optimal_batch_size():
    if DEVICE == "cuda":
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            optimal_size = max(1, int((gpu_mem / 1024**3) * 2))
            return min(optimal_size, 16)
        except:
            return 8
    else:
        system_mem = psutil.virtual_memory().available / (1024**3)
        return max(1, min(4, int(system_mem / 2)))

CHUNK_SIZE = 2000
BUFFER_SIZE = 1024 * 1024
BATCH_SIZE = get_optimal_batch_size()
print(f"Optimal batch size: {BATCH_SIZE}")

torch.set_grad_enabled(False)
dtype = torch.float16 if DEVICE == "cuda" else torch.float32
pipe = WhisperPipeline(device=DEVICE)

print(f"Running on {DEVICE}")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(ROOT_DIR, 'PDF')
TXT_DIR = os.path.join(ROOT_DIR, 'TXT')
AUDIO_DIR = os.path.join(ROOT_DIR, 'Audio')

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

def chunk_generator(text: str, chunk_size: int):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, 'rb', buffering=BUFFER_SIZE) as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() + '\n' for page in reader.pages)
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_txt(txt_path: str) -> str:
    try:
        with open(txt_path, 'r', encoding='utf-8', buffering=BUFFER_SIZE) as file:
            text = file.read()
        if not text.strip():
            raise ValueError("Text file is empty")
        return text
    except Exception as e:
        raise Exception(f"Error reading TXT: {str(e)}")

def process_text(text: str | list) -> np.ndarray | list:
    if DEVICE == "cuda":
        with torch.amp.autocast('cuda', dtype=dtype):
            if isinstance(text, list):
                results = [pipe.generate(chunk) for chunk in text]
                return [result.cpu() for result in results] if DEVICE == "cuda" else results
            result = pipe.generate(text)
            return result.cpu() if DEVICE == "cuda" else result
    
    if isinstance(text, list):
        results = [pipe.generate(chunk) for chunk in text]
        return [result.cpu() for result in results] if DEVICE == "cuda" else results
    result = pipe.generate(text)
    return result.cpu() if DEVICE == "cuda" else result

def manage_memory(audio_data=None):
    if audio_data is not None:
        del audio_data
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

def convert_text_to_speech(text: str, output_path: str):
    try:
        chunk_gen = chunk_generator(text, CHUNK_SIZE)
        audio_chunks = []
        current_batch = []
        
        max_workers = 1 if DEVICE == "cuda" else min(4, multiprocessing.cpu_count())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for chunk in chunk_gen:
                current_batch.append(chunk)
                
                if len(current_batch) >= BATCH_SIZE:
                    batch_results = process_text(current_batch)
                    audio_chunks.extend(batch_results)
                    current_batch = []
                    manage_memory()
            
            if current_batch:
                batch_results = process_text(current_batch)
                audio_chunks.extend(batch_results)
        
        audio = np.concatenate([chunk.cpu().numpy() for chunk in audio_chunks])
        manage_memory(audio_chunks)
        
        # Convert to AudioArrayClip and save as MP4 with improved quality
        audio_clip = AudioArrayClip(audio.reshape(-1, 1), fps=SAMPLE_RATE)
        audio_clip.write_audiofile(
            output_path,
            codec='aac',
            bitrate='320k',
            ffmpeg_params=[
                '-ar', str(SAMPLE_RATE),
                '-ac', '2',  # Stereo output
                '-b:a', '320k',  # High bitrate
                '-q:a', '0',  # Highest quality
            ]
        )
        audio_clip.close()
        
        manage_memory(audio)
            
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        raise

def process_file(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + '.mp4'  # Changed to .mp4
        output_path = os.path.join(AUDIO_DIR, output_filename)
        
        print("Extracting text from file...")
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        print("Converting text to speech...")
        convert_text_to_speech(text, output_path)
        
        print(f"Audio file saved as: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def process_all_files():
    for directory, extensions in [(PDF_DIR, '.pdf'), (TXT_DIR, '.txt')]:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        
        files = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
        
        if not files:
            print(f"No {extensions} files found in {directory}")
            continue
        
        for file in files:
            file_path = os.path.join(directory, file)
            print(f"\nProcessing: {file}")
            process_file(file_path)

if __name__ == "__main__":
    process_all_files()