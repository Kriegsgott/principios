import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def display_fft(signal, fs):
    N = np.size(signal)
    # sample spacing
    T = 1.0 / fs
    x = np.linspace(0.0, N * T, N)
    yf = fft(signal)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.title('FFT señal a transmitir')
    plt.xlabel('Frecuencia HZ')
    plt.grid()
    plt.show()
