# -*- coding: utf-8 -*-
"""
Contains the functions to modulate and demodulate with 16QAM

@author: Hojin
"""

from audio_play import *
from process_word import *
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from display import display_fft


def qam16(bitwise_representation, f, fs, periods=20):
    """
    Does the 16QAM modulation
    
    :param bitwise_representation:  What we want to transmit grouped in 4 bits
    :param f:                       The frequency of the Carrier Wave
    :param fs:                      Sampling frequency
    :param periods:                 Number of periods per bit
    :return:                        A numpy array containing the wave
    """
    # Mask that filters the first 2 bits of the number
    mask = (1 << 2) - 1
    I_components = np.array([])
    Q_components = np.array([])
    
    # Loop over the numbers
    for number in bitwise_representation:
        # I and Q components of the wave
        I_component = coding_values(int(number & mask) - 2)
        Q_component = coding_values(int((number >> 2) & mask) - 2)
        Ts = 1/fs

        I_components = np.append(I_components, np.ones(int(periods))*I_component)
        Q_components = np.append(Q_components, np.ones(int(periods))*Q_component)

    # Parametros del Butterworth
    fc = 700
    w = fc / (fs / 2)
    order = 5

    # Filter the signal
    b, a = signal.butter(order, w, 'low')
    I_components = signal.filtfilt(b, a, I_components)
    Q_components = signal.filtfilt(b, a, Q_components)

    # Add the wave to the list
    waves = (I_components*generateWave(f, fs, Ts*np.size(I_components), val=1) -
                 Q_components*generateWave(f, fs, Ts*np.size(Q_components), val=0))

    return waves
    

def qam16_demodulate(waves, f, fs):
    """
    Does the 16QAM demodulation.
    
    :param waves:   Numpy array corresponding to the received wave
    :param f:       Carrier frequency
    :param fs:      Sampling frequency
    :return:        A tuple containing the I and Q components
    """
    cosine_wave = generateWave(f, fs, np.size(waves)/fs, val=1)[:np.size(waves)]
    sine_wave = generateWave(f, fs, np.size(waves)/fs, val=0)[:np.size(waves)]
    
    # Componentes I y Q
    I_component = 2*waves*cosine_wave
    Q_component = 2*waves*sine_wave

    # Parametros del Butterworth
    fc = 700
    w = fc / (fs / 2)
    order = 5

    # Filter the signal
    b, a = signal.butter(order, w, 'low')
    I_component = signal.filtfilt(b, a, I_component)
    Q_component = signal.filtfilt(b, a, Q_component)

    plt.clf()
    plt.plot(I_component[:12000])
    plt.show()

    return (I_component, Q_component)


def get_values(component, samples):
    """
    Get the values of the I and Q component.
    
    :param component:   The numpy array of either component
    :param samples:     The number of samples in each component
    :return:            A list with the components actual values
    """
    # Size of the component
    size = np.size(component)
    
    # Values to be returned
    values = []
    value = samples/2

    # Iterate over the array
    while True:
        # If we are done, break
        if value > size:
            break
        
        # Else we append the value
        values.append(approximate(np.mean(component[int(value - samples/5):
                                                    int(value + samples/5)])))
        value += samples
    
    return values


def approximate(value):
    """
    Approximates the I or Q component
    
    :param value:   The value to approximate
    :return:        The approximated value
    """
    
    # If less than 1 it's actually 0
    if abs(value) < 1:
        return_value = 0
        
    # If positive it is either 2 or 4
    elif value > 0:
        return_value = 2 if abs(value - 2) < abs(value - 4) else 4
    
    # If negative it is either -2 or -4
    else:
        return_value = -2 if abs(value + 2) < abs(value + 4) else -4
        
    return return_value


def coding_values(component):
    """
    Code the I and Q components
    
    :param component:   Either the I or Q component
    :return:            The component with the operation done
    """
    # Adjust the I and Q components (we don't want zero and want 2s and 4s)
    component = component + 1 if (component >= 0) else component
    component *= 2
    
    return component


def reverse(component):
    """
    Reverse the operations in I and Q components
    
    :param component:   Either an I or Q component
    :return:            The component with the reverse operation done
    """
    
    component /= 2
    component = component - 1 if (component > 0) else component
    component += 2
    
    return component


def group_fours(numbers):
    """
    Groups the 8 bit chars in groups of 4 bits

    :param numbers:  List of numbers corresponding to the characters ASCII values
    :return:         A List containing 4 bit numbers corresponding to the
    received 8 bit numbers.
    """
    # New numbers to be generated
    new_numbers = []
    
    # Mask used to get the 4 first numbers
    mask = (1 << 4) - 1
    
    # Loop over the numbers received
    for number in numbers:
        new_numbers.append(number & mask)
        new_numbers.append((number >> 4) & mask)
    
    return new_numbers        


def group_eights(numbers):
    """
    Groups the numbers to eights from the groups of fours
    
    :param numbers:     The list of numbers grouped in fours
    :return:            The list of numbers grouped in eights
    """
    values = []
    # Loop over the numbers
    for i in range(int(len(numbers)/2)):
        # Get the original 8 bit value
        values.append(numbers[2*i] | (numbers[2*i + 1] << 4))

    return values


def decode_components(I, Q):
    """
    Get the values from the I and Q components
    
    :param I:   A list with the I components
    :param Q:   A list with the Q components
    :return:    A list with the values from the I and Q components
    """
    
    # Values obatined from I and Q components
    values = []
    
    # Loop over the I and Q components
    for i in range(len(I)):
        # If either component is 0 we have a invalid value
        if I[i] == 0 or Q[i] == 0:
            continue
        
        # Else get the original value
        I_component = reverse(I[i])
        Q_component = reverse(Q[i])
        
        # Get the original value
        value = int(I_component) | (int(Q_component) << 2)
        values.append(value)
        
    return values


def synchronizer_signal(periods):
    """
    Generates the signal used to synchronize the receiver
    
    :param periods:     Number of periods for the synchronizer signal
    :return:            A list with the numbers grouped in fours
    """
    
    numbers = []
    
    # Append numbers to the list
    for i in range(periods):
        
        # 15 = 00001111
        numbers.append(15)
        
    return group_fours(numbers)


def estimator_signal(periods):
    """
    Generates the signal used to estimate the points of the 16QAM
    
    :param periods:     Number of periods for the estimator signal
    :return:            A list with the numbers grouped in fours
    """
    
    numbers = []
    
    # append numbers to the list
    for i in range(periods):
        # 15 = 00001111
        numbers.append(15)
        # 165 = 10100101
        numbers.append(165)
        
    return group_fours(numbers)
