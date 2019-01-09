#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the functions needed to generate a wave and play it

@author: hojin
"""

import pyaudio
import numpy as np

def generateWave(f, fs, duration, val = 0):
    """
    Generates a wave with frequency f.
    
    :param f:           Frequency of the wave
    :param fs:          Sampling frequncy
    :param duration:    Duration of the wave
    :param val:         Generates sine wave if 0, cos wave if 1
    :return:            The sampled wave
    """
    if val == 0:
        samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
    else:
        samples = (np.cos(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

    return samples


def playAudio(sound, fs, volume):
    """
    Plays a sound.
    
    :param sound:       A numpy array corresponding to the sound to be played
    :param fs:          Sampling frequency
    :param volume:      A value in [0, 1] corresponding to the volume
    """
    p = pyaudio.PyAudio()
    
    # Generate the stream
    stream = p.open(format=pyaudio.paFloat32, channels=1,
                    rate=fs, output=True)
    
    # Play
    stream.write(volume*sound)
    
    # Finish
    stream.stop_stream()
    stream.close()
    p.terminate()