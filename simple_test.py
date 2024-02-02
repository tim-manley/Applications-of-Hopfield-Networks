import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import network
from scipy.io import wavfile

# Utils
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(5, 6)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

def preprocessing(img, w=128, h=128):
    # Resize image
    img = resize(img, (w,h), mode='reflect')

    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int

    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def main():

    # Read the wav file
    sample_rate, hz440 = wavfile.read('./audio/Sine_wave_440.wav')
    _, hz250 = wavfile.read('./audio/Sine_wave_250.wav')
    _, hz10k = wavfile.read('./audio/Sine_wave_10k.wav')
    
    # Take the fourier transform of the wav file, reduce the size for memory efficiency TODO: figure out optimal n
    fft_result_440 = np.fft.fft(hz440, len(hz440)//64)
    fft_result_250 = np.fft.fft(hz250, len(hz250)//64)
    fft_result_10k = np.fft.fft(hz10k, len(hz10k)//64)

    # Frequencies corresponding to the FFT result
    frequencies = np.fft.fftfreq(len(fft_result_440), 1/sample_rate)

    # Just the positive parts
    positive_fft_440 = np.abs(fft_result_440)[:len(frequencies)//2]
    positive_fft_250 = np.abs(fft_result_250)[:len(frequencies)//2]
    positive_fft_10k = np.abs(fft_result_10k)[:len(frequencies)//2]

    # Marge data
    data = [positive_fft_250, positive_fft_440, positive_fft_10k]

    # Preprocessing
    print("Start to data preprocessing...")
    data = [preprocessing(d) for d in data]

    # Create Hopfield Network Model
    model = network.HopfieldNetwork()
    model.train_weights(data)

    # Generate testset
    test = [get_corrupted_input(d, 0.3) for d in data]

    predicted = model.predict(test, threshold=0, asyn=False)
    print("Show prediction results...")
    plot(data, test, predicted)
    print("Show network weights matrix...")
    #model.plot_weights()

if __name__ == '__main__':
    main()