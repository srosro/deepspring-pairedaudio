import os
import numpy as np
from scipy.io import wavfile
import noisereduce as nr

# Recording directories
RECORDINGS_DIR = os.path.expanduser("~/recordings/denoised")
PROCESSED_DIR = os.path.expanduser("~/recordings/noisereduced")

def process_audio_file(input_file, output_file):
    # Read the raw recording
    sample_rate, audio_data = wavfile.read(input_file)

    ## Ensure audio_data is in the correct format (float32)
    #audio_data = audio_data.astype(np.float32)

    # If stereo, convert to mono by averaging channels
    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)

    # Use n_fft=512 for speech processing (23ms at 22050 Hz)
    reduced_noise = nr.reduce_noise(
        y=audio_data, 
        sr=sample_rate,
        n_jobs=1,
        n_fft=512,
        stationary=False,
        prop_decrease=0.75,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        thresh_n_mult_nonstationary=2,
        sigmoid_slope_nonstationary=10,
        use_tqdm=True,
    )

    ## Convert back to int16 for saving
    #reduced_noise = (reduced_noise * 32767).astype(np.int16)

    # Save the processed recording as a new WAV file
    wavfile.write(output_file, sample_rate, reduced_noise)

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
