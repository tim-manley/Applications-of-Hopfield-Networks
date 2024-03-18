import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import normalize
from pynufft import NUFFT
import matplotlib.pyplot as plt

def mags_to_wav(output_filename, magnitudes, duration, sample_rate):

    num_samples = int(duration * sample_rate)

    #complex_signal = magnitudes * np.exp(1j * np.random.uniform(0, 2*np.pi, len(magnitudes)))
    complex_signal = magnitudes #* np.exp(1j * np.random.uniform(0, 2*np.pi, len(magnitudes)))

    # Step 3: Synthesize Time-domain Signal
    time_domain_signal = np.fft.irfft(complex_signal)

    # Step 4: Normalization
    time_domain_signal /= np.max(np.abs(time_domain_signal))

    original_length = len(time_domain_signal)

    
    # Repeat for desired duration
    reps = num_samples // len(time_domain_signal) + 1
    time_domain_signal = np.tile(time_domain_signal, reps)[:num_samples]

    # Apply fade-in effect to the beginning of each repetition
    fade_length = int(sample_rate * 0.0000001)  # adjust the fade-in length as needed
    fade_in_curve = np.linspace(0, 1, fade_length)
    fade_out_curve = np.linspace(1, 0, fade_length)
    gain = np.minimum(np.linspace(0, 1, fade_length), np.linspace(1, 0, fade_length))
    for i in range(reps):
        start_index = i * fade_length
        end_index = start_index + fade_length
        
        # Calculate gain adjustment for crossfade
        gain = np.minimum(np.linspace(0, 1, fade_length), np.linspace(1, 0, fade_length))
        
        # Apply crossfading with gain adjustment
        time_domain_signal[start_index:start_index + fade_length] *= gain
        time_domain_signal[start_index:start_index + fade_length] += time_domain_signal[end_index:end_index + fade_length] * (1 - gain)


    # Step 5: Convert to WAV
    wavfile.write(output_filename, sample_rate, time_domain_signal.astype(np.float32))

def mel_scale(frequencies, num_mel_filters, sample_rate):
    """
    Convert linear frequency scale to Mel scale.

    Parameters:
        frequencies (array-like): Array of frequencies in Hz.
        num_mel_filters (int): Number of Mel filters.
        sample_rate (int): Sampling rate of the signal.

    Returns:
        Array of frequencies converted to Mel scale.
    """
    # Convert frequencies to Mel scale using the formula: mel(f) = 2595 * log10(1 + f/700)
    # Douglas O'Shaughnessy (1987). Speech communication: human and machine
    mel_frequencies = 2595 * np.log10(1 + frequencies / 700)
    
    # Create equally spaced points in Mel scale
    mel_points = np.linspace(mel_frequencies[0], mel_frequencies[-1], num_mel_filters + 2)
    
    # Convert Mel scale points back to linear scale
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    
    return hz_points

def compute_mel_spectrogram(wave_file, num_mel_filters=40):
    """
    Compute Mel spectrogram from a wave file.

    Parameters:
        wave_file (str): Path to the wave file.
        num_mel_filters (int): Number of Mel filters.

    Returns:
        Mel spectrogram.
    """
    # Read the wave file
    sample_rate, signal = wavfile.read(wave_file)
    
    # Compute the short-time Fourier transform (STFT)
    stft = np.abs(np.fft.fft(signal))
    
    # Define the frequency axis
    freq_axis = np.fft.fftfreq(len(signal), d=1/sample_rate)

    # Only take positive frequencies
    freq_axis = freq_axis[:len(freq_axis)//2]
    stft = stft[:len(freq_axis)]

    # Pre-normalize
    print(stft.shape)
    stft = normalize(stft.reshape(-1, 1), axis=0)
    print(stft)
    
    # Convert frequencies to Mel scale
    mel_freqs = mel_scale(freq_axis, num_mel_filters, sample_rate)
    
    # Initialize mel spectrogram
    mel_spectrogram = np.zeros(num_mel_filters)
    
    # Compute Mel spectrogram
    for i in range(num_mel_filters):
        # Find indices of the STFT frequencies that fall within the current Mel filter
        indices = np.where(np.logical_and(freq_axis >= mel_freqs[i], freq_axis <= mel_freqs[i+2]))[0]
        '''print("--------")
        print(i)
        print(freq_axis[min(indices)])
        print(freq_axis[max(indices)])
        print("Highest mag in range: ", max(stft[indices]))'''
        
        # Take mean of energy in filter range (accounts for increased scaling)
        mel_spectrogram[i] = np.mean(stft[indices], axis=0)
        '''print(f"Mel frequency bin: [{mel_freqs[i]}, {mel_freqs[i+2]}]")
        print("Mel mag: ", mel_spectrogram[i])'''
    
    # Return the magnitudes and corresponding mel frequencies (midpoints)
    return mel_spectrogram, mel_freqs[1:len(mel_freqs)-1]

def mel_to_wave(mel_spectrogram, mel_freqs):
    """
    Reconstruct wave file from Mel spectrogram and frequencies.

    Parameters:
        mel_spectrogram (2D array-like): Mel spectrogram.
        mel_frequencies (array-like): Array of frequencies in mel scale.

    Returns:
        Reconstructed wave signal.
    """

    # Convert Mel scale to linear scale
    # TODO: Figure out optimal number of samples...
    freq_axis = np.linspace(mel_freqs[0], mel_freqs[-1], len(mel_spectrogram))
    
    # Initialize signal
    signal = np.zeros(len(freq_axis))
    
    # Reconstruct signal from Mel spectrogram
    for i in range(len(mel_spectrogram) - 2):
        # Find indices of the frequencies that fall within the current Mel filter
        indices = np.where(np.logical_and(freq_axis >= mel_freqs[i], freq_axis <= mel_freqs[i+2]))[0]
        
        # Distribute energy to the corresponding frequencies within the filter
        signal[indices] += mel_spectrogram[i]
    
    # Take the inverse Fourier transform to get time-domain signal
    reconstructed_signal = np.fft.irfft(signal)

    reconstructed_signal /= np.max(np.abs(reconstructed_signal))

    # Scale the signal to the appropriate range for 16-bit PCM WAV files
    scaled_signal = np.int16(reconstructed_signal * 32767)
    
    return scaled_signal

# Helper function linearly maps number from [0,1] to an n-bit number in [0,2^n - 1]
def float_to_bin_array(num, nbits):
    # check data is normalized
    if num < 0 or num > 1:
        raise ValueError("Input value must be in the range [0,1]")
    binary_array = np.binary_repr(int(num * (2 ** nbits - 1)), width=nbits)
    binary_array = np.array(list(map(np.int8, binary_array)))
    binary_array[binary_array == 0] = -1
    return binary_array

def binarize_audio(magnitude_data, nbits):
    bin_data = []
    for d in magnitude_data:
        bin_arr = float_to_bin_array(d, nbits)
        bin_data.append(bin_arr)
    bin_data = np.array(bin_data)
    return bin_data

def unbinarize_data(binarized_data):
    original_mags = []
    data = np.array(binarized_data, dtype=np.int8)
    data[data == -1] = 0
    data = np.array(data, dtype=np.uint8)
    nbits = len(data[0])
    for d in data:
        # Convert the binary array to a packed binary representation
        # Pad with leading 0s until multiple of 4
        num_to_pad = 8 - (nbits % 8)
        if num_to_pad != 8:
            d = np.insert(d, 0, np.array([0] * num_to_pad, dtype=np.int8))
        packed_binary = np.packbits(d)
        packed_binary = np.array(packed_binary, dtype=np.float64)
        packed_binary /= 2 ** nbits - 1
        # Unpack the binary representation into a float64 value
        original_mags.append(packed_binary)
    return np.array(original_mags).squeeze()

def add_noise_to_wav(wav_file, output_file, noise_level=0.1):
    # Load the WAV file
    sample_rate, data = wavfile.read(wav_file)
    
    # Generate Gaussian noise with the same length as the audio data
    noise = np.random.normal(0, noise_level, data.shape)
    
    # Add noise to the audio data
    noisy_data = np.add(data, noise)
    
    # Ensure data is within the valid range for int16 (for 16-bit WAV files)
    # Clip values to stay within the valid range [-32768, 32767]
    noisy_data = np.clip(noisy_data, -32768, 32767).astype(np.int16)
    
    # Write the noisy audio data to a new WAV file
    wavfile.write(output_file, sample_rate, noisy_data)

def generate_sine_wave_file(file_name, frequency=440, duration=1, sample_rate=44100, amplitude=32767):
    """
    Generate a sine wave and save it as a WAV file using scipy.io.wavfile.write.
    
    Parameters:
        file_name (str): Name of the output WAV file.
        frequency (float): Frequency of the sine wave in Hz.
        duration (float): Duration of the sine wave in seconds.
        sample_rate (int): Sampling rate (number of samples per second).
        amplitude (int): Amplitude of the sine wave.
    """
    time = np.linspace(0, duration, sample_rate * duration)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)

    # Scale to the range of int16 (which is what write expects)
    scaled_wave = np.int16(sine_wave)

    # Save the wave file
    wavfile.write(file_name, sample_rate, scaled_wave)

if __name__ == "__main__":
    test_data = np.array([0.5, 0.25, 0.7, 0])

    binarized = binarize_audio(test_data, 16)

    print(binarized)

    unbinarized = unbinarize_data(binarized)
    
    print(unbinarized)
