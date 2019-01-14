# -*- coding: utf-8 -*-
"""
Test the transmitter module

@author: Hojin
"""

from transmitter import generate_signal
from receiver import decode_signal
from display import display_fft

import scipy.io.wavfile as wav
import numpy as np

# Word and image
word = "¡Examen, Principio de Comunicaciones Primavera 2018 EL4005!"
image_path = "pollito_14x14.png"

# Frequencies used
f_word = 4000
f_R = 6000
f_G = 8000
f_B = 10000
fs = 40000

# Periods per bit
periods = 2000

# Generate signal
signal = generate_signal(word, image_path, f_word, f_R, f_G, f_B, fs, periods,
                    repetitions=3, sync_repetitions=3, estimation_repetitions=3,
                    inter_repetition_periods=1)

signal = np.append(np.zeros(800), signal)

# Noise
noise = np.random.normal(0, 1.0, np.size(signal))
signal += noise
scaled = np.int16(signal/np.max(np.abs(signal)) * 32767)
display_fft(signal, fs)
wav.write('test.wav', 44100, scaled)

word = decode_signal(signal, f_word, f_R, f_G, f_B, fs)

