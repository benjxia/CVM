import simpleaudio as sa
from scipy import io
import numpy as np

import sys
import argparse

if __name__ == '__main__':
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    filename = 'Recording.wav'

    sample_rate, audio_data = io.wavfile.read(filename)
    audio_buffer: np.ndarray = audio_data.astype(np.int16)

    play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate * 2)

    play_obj.wait_done()
