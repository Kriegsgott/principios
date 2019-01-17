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


scaled = np.float32((signal/np.max(np.abs(signal))))
wav.write('test.wav', 44100, scaled)
fs1, test = wav.read(r'C:\Users\dudam\PycharmProjects\principios\grabacion2.wav')

word = decode_signal(test, f_word, f_R, f_G, f_B, fs)

