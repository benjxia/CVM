import simpleaudio as sa
from scipy import io
from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans
import numpy as np

import sys
import argparse

from pydub import AudioSegment, effects
import librosa

"""
Program specs
Ability to change sample rate
Change pitch
Bass boost 
"""


# Shitty bass boosting from chatgpt lmao
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bass_boost(audio_np_array, sample_rate, bass_gain=6, cutoff_frequency=200):
    # Apply a low-pass filter to boost bass frequencies
    filtered_audio = butter_lowpass_filter(audio_np_array, cutoff_frequency, sample_rate)

    # Increase the amplitude of filtered audio to boost bass
    boosted_audio = filtered_audio * (10 ** (bass_gain / 20))

    return boosted_audio.astype(np.int16)

def apply_kmeans(audio_np_array, num_clusters=2):
    # Reshape audio data to 2D array for clustering
    data_2d = audio_np_array.reshape(-1, 1)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data_2d)

    # Assign each sample to its cluster center
    clustered_data = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape back to original shape
    clustered_audio = clustered_data.reshape(-1)

    return clustered_audio.astype(np.int16)

def change_speed(input_file, playback_rate):
    audio = AudioSegment.from_file(input_file)

    changed_audio = audio.speedup(playback_speed=playback_rate)

    audio_data = changed_audio.raw_data
    sample_width = changed_audio.sample_width
    channels = changed_audio.channels
    frame_rate = changed_audio.frame_rate

    playback_obj = sa.play_buffer(audio_data, num_channels=channels, bytes_per_sample=sample_width, sample_rate=frame_rate)

    playback_obj.wait_done()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', type=str, help='Specify the file name.', required=True)
    parser.add_argument('-b', '--bass', type=float, help='Specify the constant to alter the bass of the track.')
    parser.add_argument('-s', '--speed', type=float, help='Specify the scalar constant to change the speed of the track.')
    parser.add_argument('-p', '--pitch', type=float, help='Specify the constant to shift the pitch of the track.')
    parser.add_argument('-d', '--deepfry', type=int, help='Specify the constant for deep fried mic.')

    args = parser.parse_args()

    sample_rate, audio_data = io.wavfile.read(args.filename)
    audio_buffer: np.ndarray = audio_data.astype(np.int16)

    if args.bass is not None:
        audio_buffer = bass_boost(audio_buffer, sample_rate, bass_gain=args.bass, cutoff_frequency=200)
        play_obj = sa.play_buffer(audio_buffer, 1, 2, sample_rate)
        play_obj.wait_done()
    if args.speed is not None:
        change_speed(args.filename, args.speed)
    #if args.pitch is not None:
        # implement pitch shift
    if args.deepfry is not None:
        audio_buffer = apply_kmeans(audio_buffer, num_clusters=args.deepfry)
        play_obj = sa.play_buffer(audio_buffer, 1, 2, sample_rate)
        play_obj.wait_done()

