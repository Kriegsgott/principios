# -*- coding: utf-8 -*-
"""
Test the transmitter module

@author: Hojin
"""

from Tools.transmitter import generate_signal
from Tools.audio_play import playAudio
from Tools.receiver import decode_signal

import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt

# Word and image
word = "¡Examen, Principio de Comunicaciones Primavera 2018 EL4005!"
image_path = "pollito_14x14.png"

# Frequencies used
f_word = 1000
f_R = 5000
f_G = 9000
f_B = 13000
fs = 40000

# Periods per bit
periods = 200

# Generate signal
signal = generate_signal(word, image_path, f_word, f_R, f_G, f_B, fs, periods,
                    repetitions=3, sync_repetitions=3, estimation_repetitions=3,
                    inter_repetition_periods=1)

signal = np.append(np.zeros(800), signal)

scaled = np.int16(signal/np.max(np.abs(signal)) * 32767)
wav.write('test.wav', 44100, scaled)

word = decode_signal(scaled, f_word, f_R, f_G, f_B, fs)

