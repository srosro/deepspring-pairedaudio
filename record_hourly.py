import pyaudio
import wave
import schedule
import time
import os
from datetime import datetime
from pytz import timezone
import numpy as np
import noisereduce as nr
from scipy import signal

# Directory to save recordings
OUTPUT_DIR = os.path.expanduser("~/recordings")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3600  # 1 hour

def record_audio():
    # Get the current time in PST
    pst = timezone('America/Los_Angeles')
    timestamp = datetime.now(pst).strftime('%Y-%m-%d_%H-%M-%S')
    raw_output_file = os.path.join(OUTPUT_DIR, f"raw_recording_{timestamp}.wav")
    final_output_file = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.wav")

    audio = pyaudio.PyAudio()

    # Open the audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print(f"Recording started: {raw_output_file}")

    # Open the raw WAV file for writing
    with wave.open(raw_output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        # Record for the specified number of seconds
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            wf.writeframes(data)

    print("Recording finished")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Read the raw recording and apply audio processing
    with wave.open(raw_output_file, 'rb') as wf:
        raw_audio_data = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(raw_audio_data, dtype=np.int16)
        
        # Convert to float32 for processing
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=audio_float, sr=RATE)

        # Apply a high-pass filter to remove low-frequency noise
        sos = signal.butter(10, 100, 'hp', fs=RATE, output='sos')
        filtered_audio = signal.sosfilt(sos, reduced_noise)

        # Apply dynamic range compression
        threshold = 0.1
        ratio = 4.0
        compressed_audio = np.where(
            np.abs(filtered_audio) > threshold,
            threshold + (np.abs(filtered_audio) - threshold) / ratio * np.sign(filtered_audio),
            filtered_audio
        )

        # Normalize the audio
        normalized_audio = np.int16(compressed_audio / np.max(np.abs(compressed_audio)) * 32767)

    # Save the processed recording as a new WAV file
    with wave.open(final_output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(normalized_audio.tobytes())

    # Optionally, remove the raw recording file to save space
    os.remove(raw_output_file)

def schedule_recording():
    # Schedule the recording to start at the beginning of every hour
    schedule.every().hour.at(":00").do(record_audio)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    schedule_recording()
