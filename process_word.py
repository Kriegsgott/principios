# -*- coding: utf-8 -*-
"""
Contains the functions needed to generates the bits from a word

@author: Hojin
"""


def generate_stream(word):
    """
    Generates a stream of binary numbers representing the letters of the word
    
    :param word:    Word to generate stream from
    :return:        A list with the numbers in binary form
    """
    # List of numbers to transmit
    numbers = []
    
    # For loop over each letter of the word
    for char in word:
        numbers.append(ord(char))
        
    return numbers


def generate_original_word(numbers):
    """
    Generates the ASCII word from the numbers given
    
    :param numbers:     The list of numbers
    :return:            The ASCII word
    """
    word = ''
    for number in numbers:
        word += chr(number)
        
    return word
