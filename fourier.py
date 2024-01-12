from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

def preprocess_wavfile(filepath):
    # Read the WAV file
    sample_rate, data = wavfile.read(filepath)

    # Take the fourier transform
    fft_result = np.fft.fft(data)

    # Return the sample_rate and the fft_result
    return sample_rate, fft_result

preprocess_wavfile('./audio/Sine_wave_440.wav')

'''
# Read the WAV file
sample_rate, data = wavfile.read('./audio/Sine_wave_440.wav')

# Take the Fourier transform
fft_result = np.fft.fft(data)

# Frequencies corresponding to the FFT result
frequencies = np.fft.fftfreq(len(fft_result), 1/sample_rate)

# Plot the magnitude spectrum
plt.figure(figsize=(8, 4))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.show()
'''