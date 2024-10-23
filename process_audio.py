import os
import wave
import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile
from scipy.signal import medfilt
from pydub import AudioSegment

# Recording directories
RECORDINGS_DIR = os.path.expanduser("~/recordings/raw")
PROCESSED_DIR = os.path.expanduser("~/recordings/denoised")

def process_audio_file(input_file, output_file):
    # Read the raw recording and apply audio processing
    rate, data = wavfile.read(input_file)

    # Check if the audio data is empty
    if len(data) == 0:
        print(f"Warning: {input_file} is empty or corrupted. Skipping processing.")
        return

    # Determine number of channels
    channels = 2 if data.ndim == 2 else 1

    # Apply a median filter to remove clicks
    # Ensure kernel_size is odd and smaller than the data size
    kernel_size = min(3, len(data) - 1)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        filtered_data = data  # Skip median filtering if data is too short
    else:
        filtered_data = medfilt(data, kernel_size=kernel_size)

    # Convert the filtered data to an AudioSegment
    audio = AudioSegment(
        filtered_data.tobytes(),
        frame_rate=rate,
        sample_width=filtered_data.dtype.itemsize,
        channels=channels
    )

    # Export the filtered audio back to a numpy array
    filtered_data = np.array(audio.get_array_of_samples())

    print(f"Clicking sounds removed from {input_file}")

    # Convert to numpy array
    audio_data = filtered_data

    # If the audio is stereo, reshape it
    if channels == 2:
        audio_data = audio_data.reshape(-1, 2)

    # Convert to float32 for processing
    audio_float = audio_data.astype(np.float32) / 32768.0

    # Check if audio_float is empty
    if audio_float.size == 0:
        print(f"Warning: Processed audio data for {input_file} is empty. Skipping further processing.")
        return

    # Apply a high-pass filter to remove low-frequency noise
    sos = signal.butter(10, 100, 'hp', fs=rate, output='sos')
    filtered_audio = signal.sosfilt(sos, audio_float)

    # Apply dynamic range compression
    threshold = 0.1
    ratio = 4.0
    compressed_audio = np.where(
        np.abs(filtered_audio) > threshold,
        threshold + (np.abs(filtered_audio) - threshold) / ratio * np.sign(filtered_audio),
        filtered_audio
    )

    # Normalize the audio
    max_val = np.max(np.abs(compressed_audio))
    normalized_audio = compressed_audio / max_val

    # Convert back to int16
    final_audio = np.int16(normalized_audio * 32767)

    # Save the processed recording as a new WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)  # Preserve original channels
        wf.setsampwidth(2)  # 2 bytes (16 bits) per sample
        wf.setframerate(rate)
        wf.writeframes(final_audio.flatten().tobytes())

    print(f"Processed audio saved as {output_file}")

def process_recordings():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for filename in os.listdir(RECORDINGS_DIR):
        if filename.endswith(".wav"):
            input_file = os.path.join(RECORDINGS_DIR, filename)
            output_file = os.path.join(PROCESSED_DIR, f"processed_{filename}")

            if not os.path.exists(output_file):
                print(f"Processing {filename}...")
                process_audio_file(input_file, output_file)
            else:
                print(f"Processed version of {filename} already exists. Skipping.")

if __name__ == "__main__":
    process_recordings()
