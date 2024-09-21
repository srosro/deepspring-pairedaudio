import pyaudio
import wave
import schedule
import time
import os
from datetime import datetime
from pytz import timezone

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
    output_file = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.wav")

    audio = pyaudio.PyAudio()

    # Open the audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print(f"Recording started: {output_file}")
    frames = []

    # Record for the specified number of seconds
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording as a WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def schedule_recording():
    # Schedule the recording to start at the beginning of every hour
    schedule.every().hour.at(":00").do(record_audio)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    schedule_recording()
