# -*- coding: utf-8 -*-
"""
Functions needed to transmit the message

@author: Hojin
"""

from modulation import *
from process_image import *
from process_word import *

import numpy as np


def generate_signal(word, image_path, f_word, f_R, f_G, f_B, fs, periods,
                    repetitions=3, sync_repetitions=3, estimation_repetitions=3,
                    inter_repetition_periods=3):
    """
    Generates the signal to be transmitted
    
    :param word:                    Word to be transmitted
    :param image_path:              Path to the image to be transmitted
    :param f_word:                  Frequency for the carrier of the word
    :param f_R:                     Frequency of the Red Channel of the image
    :param f_G:                     Frequency of the Green Channel of the image
    :param f_B:                     Frequency of the Blue Channel of the image
    :param fs:                      Sampling Frequency
    :param periods:                 Number of periods per bit
    :param repetitions:             Number of times to repeat signal (FEC)
    :param sync_repetitions:        Number of times to reapeat sync signal
    :param estimation_repetitions:  Number of times to repeat estimation signal
    :param inter_repetition_periods:Number of periods between repetitions
    :return:                        Signal that has to be played
    """
    
    # Generate the wave for the word
    word_wave = qam16(group_fours(generate_stream(word)), f_word, fs, periods)
    
    # Generate the waves for the image
    R, G, B = generate_channels(image_path)

    R = group_fours(flatten_channel(R))
    G = group_fours(flatten_channel(G))
    B = group_fours(flatten_channel(B))
    
    R_wave = qam16(R, f_R, fs, periods)
    G_wave = qam16(G, f_G, fs, periods)
    B_wave = qam16(B, f_B, fs, periods)
    
    # Pad the shorter one with zeros
    if np.size(R_wave) > np.size(word_wave):
        word_wave_temp = word_wave
        word_wave = np.zeros(np.size(R_wave))
        word_wave[:np.size(word_wave_temp)] = word_wave_temp
    else:
        R_temp = R_wave
        G_temp = G_wave
        B_temp = B_wave
        
        R_wave = np.zeros(np.size(word_wave))
        G_wave = np.zeros(np.size(word_wave))
        B_wave = np.zeros(np.size(word_wave))
        
        R_wave[:np.size(R_temp)] = R_temp
        G_wave[:np.size(G_temp)] = G_temp
        B_wave[:np.size(B_temp)] = B_temp
    
    # Wave to transmit
    wave = word_wave + R_wave + G_wave + B_wave
    
    # Wave for sync
    sync = (qam16(synchronizer_signal(sync_repetitions), f_word, fs, periods) +
            qam16(synchronizer_signal(sync_repetitions), f_R, fs, periods) +
            qam16(synchronizer_signal(sync_repetitions), f_G, fs, periods) +
            qam16(synchronizer_signal(sync_repetitions), f_B, fs, periods))

    # Wave for estimation
    estimator = (qam16(estimator_signal(estimation_repetitions), f_word, fs, periods) +
                 qam16(estimator_signal(estimation_repetitions), f_R, fs, periods) +
                 qam16(estimator_signal(estimation_repetitions), f_G, fs, periods) +
                 qam16(estimator_signal(estimation_repetitions), f_B, fs, periods))

    # Wave for waiting
    waiting = np.zeros(int(np.size(estimator)*estimation_repetitions/
                           (inter_repetition_periods + 1e-6)))

    # Generate final wave
    final_wave = np.append(sync, estimator)

    for i in range(repetitions):
        final_wave = np.append(final_wave, wave)

    return final_wave
