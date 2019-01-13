# -*- coding: utf-8 -*-
"""
Has the functions needed in the receivew to decode the signal

@author: Hojin
"""

from modulation import *
from process_image import *
from audio_play import *
from process_word import *
import numpy as np
import matplotlib.pyplot as plt


def decode_signal(signal, f_word, f_R, f_G, f_B, fs, repetitions=3,
                  sync_repetitions=3, estimation_repetitions=3,
                  inter_repetition_periods=3):
    """
    Decodes the received signal.
    
    :param signal:                      The signal that was receives
    :param f_word:                      The frequency of the word
    :param f_R:                         The frequency of the red channel
    :param f_G:                         The frequency of the green channel
    :param f_B:                         The frequency of the blue channel
    :param fs:                          The sampling frequency
    :param repetitions:                 The number of repetitions
    :param sync_repetitions:            The number of repetitions for the sync wave
    :param estimation_repetitions:      The number of repetitions for the estimation wave
    :param inter_repetitions_periods:   The periods between repetitions
    """
    # Get image components from signal
    I_R, Q_R = qam16_demodulate(signal, f_R, fs)
    I_G, Q_G = qam16_demodulate(signal, f_G, fs)
    I_B, Q_B = qam16_demodulate(signal, f_B, fs)
    
    # Get text components from signal
    I_word, Q_word = qam16_demodulate(signal, f_word, fs)
    
    # Normalize values by 4
    I_R = I_R*4/np.max(I_R)
    Q_R = Q_R*4/np.max(Q_R)
    I_G = I_G*4/np.max(I_G)
    Q_G = Q_G*4/np.max(Q_G)
    I_B = I_B*4/np.max(I_B)
    Q_B = Q_B*4/np.max(Q_B)
    I_word = I_word*4/np.max(I_word)
    Q_word = Q_word*4/np.max(Q_word)
    
    # Get the periods of the signals
    periods_I, periods_Q, I_start, Q_start = approximate_period(I_R, I_G, I_B, I_word,
                                 Q_R, Q_G, Q_B, Q_word, sync_repetitions)
    
    # Cast the values to ints    
    periods_I = 1600
    periods_Q = 1600
    I_start = int(I_start)
    Q_start = int(Q_start)
    
    # Estimate the values of the signal
    I_word, Q_word = estimate_signal(I_word, Q_word, I_start, Q_start,
                                      periods_I, periods_Q,
                                      estimation_repetitions)
    
    I_R, Q_R = estimate_signal(I_R, Q_R, I_start, Q_start,
                               periods_I, periods_Q,
                               estimation_repetitions)
    
    I_G, Q_G = estimate_signal(I_G, Q_G, I_start, Q_start,
                              periods_I, periods_Q,
                              estimation_repetitions)
    
    I_B, Q_B = estimate_signal(I_B, Q_B, I_start, Q_start,
                               periods_I, periods_Q,
                               estimation_repetitions)
    
    # Get the real values
    I_R_Real = get_values(I_R, periods_I)
    Q_R_Real = get_values(Q_R, periods_Q)
    I_G_Real = get_values(I_G, periods_I)
    Q_G_Real = get_values(Q_G, periods_Q)
    I_B_Real = get_values(I_B, periods_I)
    Q_B_Real = get_values(Q_B, periods_Q)
    I_word_Real = get_values(I_word, periods_I)
    Q_word_Real = get_values(Q_word, periods_Q)
    
    # Generate the image
    values_R = decode_components(I_R_Real, Q_R_Real)
    values_G = decode_components(I_G_Real, Q_G_Real)
    values_B = decode_components(I_B_Real, Q_B_Real)
    R_channel = np.asarray(group_eights(values_R))
    G_channel = np.asarray(group_eights(values_G))
    B_channel = np.asarray(group_eights(values_B))
    
    # Get each of the 3 repetitions for each color
    R_1 = R_channel[:int(np.size(R_channel)/3)]
    G_1 = G_channel[:int(np.size(G_channel)/3)]
    B_1 = B_channel[:int(np.size(B_channel)/3)]
    
    R_2 = R_channel[int(np.size(R_channel)/3):int(2*np.size(R_channel)/3)]
    G_2 = G_channel[int(np.size(G_channel)/3):int(2*np.size(G_channel)/3)]
    B_2 = B_channel[int(np.size(B_channel)/3):int(2*np.size(B_channel)/3)]
    
    R_3 = R_channel[int(2*np.size(R_channel)/3):]
    G_3 = G_channel[int(2*np.size(G_channel)/3):]
    B_3 = B_channel[int(2*np.size(B_channel)/3):]
    
    # Choose by mayority in each bit
    R_channel = (R_1|R_2) & (R_2|R_3) & (R_1|R_3)
    G_channel = (G_1|G_2) & (G_2|G_3) & (G_1|G_3)
    B_channel = (B_1|B_2) & (B_2|B_3) & (B_1|B_3)
    
    # Generate image
    image = generate_array_image(R_channel, G_channel, B_channel, 14, 14)
    
    # Generate the word
    values = np.array(decode_components(I_word_Real, Q_word_Real))
    
    # Get the 3 repetitions for the word
    values_1 = values[:int(np.size(values)/3)]
    values_2 = values[int(np.size(values)/3):int(2*np.size(values)/3)]
    values_3 = values[int(2*np.size(values)/3):]
    
    # Choose by mayority by bit
    values = (values_1|values_2) & (values_2|values_3) & (values_1|values_3)
    word = generate_original_word(group_eights(values))
    
    # Print the word and generate the image
    print(word)
    plt.imshow(image)


def estimate_signal(I, Q, I_start, Q_start, periods_I, periods_Q,
                    estimation_repetitions):
    """
    Estimates the amplitude of the signal
    
    :param I:                       The I component
    :param Q:                       The Q component
    :param I_start:                 The starting position in the I component
    :param Q_start:                 The starting position in the Q component
    :param periods_I:               The periods of the I component
    :param periods_Q:               The periods of the Q component
    :param estimation_repetitions:  The repetitions for the estimation wave
    :return:                        A tuple with the rescaled values of I and Q
    """
    # Start in the middle of the first value
    I_start += periods_I/2
    Q_start += periods_Q/2
    I_start = int(I_start)
    Q_start = int(Q_start)
    
    # The expected values
    expected_vals = [4, -4, -2, 2]
    
    for i in range(4*estimation_repetitions):
        I_val = I[I_start]
        Q_val = Q[Q_start]
        expected = expected_vals[i%4]
        
        I = I*expected/I_val
        Q = Q*expected/Q_val
        
        I_start += periods_I
        Q_start += periods_Q

    # Reposition
    I_start -= periods_I/2
    Q_start -= periods_Q/2
    
    # Cast to int
    I_start = int(I_start)
    Q_start = int(Q_start)
    
    return I[I_start:], Q[Q_start:]


def approximate_period(I_R, I_G, I_B, I_word, Q_R, Q_G, Q_B,
                       Q_word, sync_repetitions):
    """
    Approximate the period of the wave
    
    :param I_R:             The I component of the Red Channel
    :param I_G:             The I component of the Green Channel
    :param I_B:             The I component of the Blue Channel
    :param I_word:          The I component of the Word
    :param Q_R:             The Q component of the Red Channel
    :param Q_G:             The Q component of the Green Channel
    :param Q_B:             The Q component of the Blue Channel
    :param Q_word:          The Q component of the Word
    :param sync_repetitions:The number of repetitions for the sync wave
    :return:                A tuple containing the periods and the new starting positions
    """
    # Get starting positions for I and Q
    I_start = np.mean(np.array([detect_start(I_R), detect_start(I_G),
                      detect_start(I_B), detect_start(I_word)]))
    
    Q_start = np.mean(np.array([detect_start(Q_R), detect_start(Q_G),
                      detect_start(Q_B), detect_start(Q_word)]))
    
    I_change_points = []
    Q_change_points = []

    # Get the values where the change happens
    for i in range(sync_repetitions*2):
        I_change = np.mean(np.array([detect_change(I_R, I_start),
                                     detect_change(I_G, I_start),
                                     detect_change(I_B, I_start),
                                     detect_change(I_word, I_start)]))
        
        Q_change = np.mean(np.array([detect_change(Q_R, Q_start),
                                     detect_change(Q_G, Q_start),
                                     detect_change(Q_B, Q_start),
                                     detect_change(Q_word, Q_start)]))
        
        # Save points
        I_change_points.append(I_change - I_start)
        Q_change_points.append(Q_change - Q_start)
        
        # Update the starting points
        I_start = I_change
        Q_start = Q_change
    
    return (np.mean(np.array(I_change_points)),
            np.mean(np.array(Q_change_points)),
            I_start, Q_start)    


def detect_start(signal, noise_samples=1000, samples=1000, threshold=5e-3):
    """
    Detects the start of the transmission
    
    :param signal:          The received signal
    :param noise_samples:   The minimum samples of noise
    :param samples:         The samples to take when estimating the start
    :param threshold:       The minimum derivative to detect the start
    :return:                The position of the start of the tranmission
    """
    # If the mean of 100 samples if over the threshold
    for i in range(noise_samples, np.size(signal)):
        if (signal[i]*signal[i+1] < 0 and all(j > 0 for j in signal[i+1:i+samples])
            and float(np.diff(signal[i:i+2])) > threshold):
            break

    return i


def detect_change(signal, start, skip=1000):
    """
    Detects a change in sign
    
    :param signal:          The received signal
    :param start:           The starting position
    :param skip:            Skip at the start to not confuse changes in sign
    :return:                The position of the first change in sign     
    """
    start += skip
    for i in range(int(start), np.size(signal)):
        if (signal[i]*signal[i+1]) < 0:
            break
        
    return i


