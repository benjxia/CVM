from scipy import io, signal
from scipy.signal import butter, lfilter, get_window
from sklearn.cluster import KMeans
import numpy as np

import simpleaudio as sa
import argparse
import librosa
from pydub import AudioSegment

import psola

"""
Program specs
Ability to change sample rate
Change pitch
Bass boost 
"""


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

    # Apply K-means clustering for audio quantization - essentially restricts audio intensities to several discrete values.
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data_2d)

    # Assign each sample to its cluster center
    clustered_data = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape back to original shape
    clustered_audio = clustered_data.reshape(-1)

    return clustered_audio.astype(np.int16)


def change_speed(audio_np_array, playback_rate):
    audio = AudioSegment.from_file(audio_np_array)

    changed_audio = audio.speedup(playback_speed=playback_rate)

    return np.array(changed_audio.get_array_of_samples())


def pitch_shift(audio_np_array, pitch_shift, sample_rate):
    fft_size = 2048 # Fast-Fourier Transform size
    hop_size = int(fft_size / 4) # Hop size should be 1/4 FFT size
    window = get_window('hann', fft_size, fftbins=True) # Window function
    num_frames = int(np.ceil(len(audio_np_array) / hop_size)) # Number of frames
    shifted_audio = np.zeros(len(audio_np_array), dtype=np.float64)

    for i in range(num_frames):
        # Get current frame boundary/Extract frame
        start = i * hop_size
        end = min(len(audio_np_array), start + fft_size)
        frame = np.zeros(fft_size)
        frame[:end - start] = audio_np_array[start:end] * window[:end - start]

        spectrum = np.fft.fft(frame)
        shift_factor = 2 ** (pitch_shift / 12)  # Convert semitones to a frequency ratio
        spectrum_shifted = np.roll(spectrum, int((fft_size / 2) * (shift_factor - 1)))
        frame_shifted = np.real(np.fft.ifft(spectrum_shifted)) # Invert fft
        frame_shifted = frame_shifted[:hop_size] # Fix frame length to hop size

        # Overlap-add
        shifted_audio[start:start + hop_size] += frame_shifted * window[:hop_size]

    # Convert the shifted_audio back to standard int16 format
    shifted_audio = shifted_audio.astype(np.int16)

    return shifted_audio

def autotune(audio_data_float, sr, scale='C:maj'):
    # autotune function
    # track pitch
    frame_length = 2048
    hop_length  = frame_length // 4
    fmin = librosa.note_to_hz('C')
    fmax = librosa.note_to_hz('C7')
    f0, _, _ = librosa.pyin(audio_data_float,
                            frame_length=frame_length,
                            hop_length=hop_length,
                            sr=sr,
                            fmin=fmin,
                            fmax=fmax)

    # calculate desired pitch
    # correct_pitch function
    corrected_f0 = np.zeros_like(f0)
    corrected_f0[np.isnan(corrected_f0)] = np.nan
    degrees = librosa.key_to_degrees(scale)
    degrees = np.concatenate((degrees, [degrees[0] + 12]))

    # Convert frequency into a note for midi purposes and set pitch to that of nearest note in user specified scale :)
    midi_note = librosa.hz_to_midi(f0)
    degree = midi_note % 12
    closest_degree_id = np.argmin(np.abs(degrees[None, :] - degree[:, None]), axis=1)
    degree_difference = degree - degrees[closest_degree_id]
    # set pitch of current snippet to nearest note
    midi_note -= degree_difference
    corrected_f0[~np.isnan(f0)] = librosa.midi_to_hz(midi_note)[~np.isnan(f0)]

    smoothed_corrected_f0 = signal.medfilt(corrected_f0, kernel_size=11)

    smoothed_corrected_f0[np.isnan(smoothed_corrected_f0)] = corrected_f0[np.isnan(smoothed_corrected_f0)]

    audio_data = psola.vocode(audio_data_float, sample_rate=int(sr), target_pitch=smoothed_corrected_f0, fmin=fmin, fmax=fmax)
    # pitch shifting
    return (audio_data * 32768).astype(np.int16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', type=str, help='Specify the file name. Must be a .wav file.', required=True)
    parser.add_argument('-a', '--autotune', type=str, help='Specify scale to tune your voice against. Should be in the form TONIC:key (e.g. C:maj for C major, C:min for C minor)')
    parser.add_argument('-b', '--bass', type=float,
                        help='Specify the constant to alter the bass of the track. Max: 50 to protect your own ears.')
    parser.add_argument('-s', '--speed', type=float,
                        help='Specify the scalar constant to change the speed of the track.')
    parser.add_argument('-p', '--pitch', type=float, help='Specify the constant to shift the pitch of the track.')
    parser.add_argument('-d', '--deepfry', type=int, help='Specify the constant for deep fried mic.')
    parser.add_argument('-o', '--out', type=str, help='Specify filename to save edited audio with.')
    args: argparse.Namespace = parser.parse_args()

    # Load audio file from
    sample_rate, audio_data = io.wavfile.read(args.filename)

    # Convert to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    audio_buffer: np.ndarray = audio_data.astype(np.int16)

    # Set file to save edited audio to. cvm.wav by default
    if args.out is not None:
        outfile = args.out
    else:
        outfile = f'cvm.wav'

    if args.speed is not None:
        if args.speed <= 0:
            raise ValueError("Speed multiplier must be greater than 0. Stop trying to time travel.")
        audio_buffer = change_speed(args.filename, args.speed)

    if args.bass is not None:
        if args.bass < 0:
            raise ValueError("Bass value must be at least 0.")

        audio_buffer = bass_boost(audio_buffer, sample_rate, bass_gain=min(args.bass, 50), cutoff_frequency=100)

    if args.pitch is not None:
        audio_buffer = librosa.effects.pitch_shift(audio_buffer.astype(np.float32), sr=sample_rate,
                                                   n_steps=args.pitch).astype(np.int16)

    if args.deepfry is not None:
        audio_buffer = apply_kmeans(audio_buffer, num_clusters=args.deepfry)

    if args.autotune is not None:
        audio_data_float = librosa.util.buf_to_float(audio_buffer)
        audio_buffer = autotune(audio_data_float, sample_rate, args.autotune).astype(np.int16)

    print("Playing audio...")
    play_obj = sa.play_buffer(audio_buffer, 1, 2, sample_rate)
    play_obj.wait_done()

    io.wavfile.write(outfile, sample_rate, audio_buffer)
