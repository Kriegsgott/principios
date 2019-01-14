# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 02:09:08 2018

@author: Hojin
"""

from PIL import Image
import numpy as np


def generate_channels(path):
    """
    Get RGB arrays from image
    
    :param path:    String corresponding to the path to the image
    :return:        A tuple with the RGB channels of the image
    """
    # Abrir imagen y transformar a array
    image = Image.open(path)
    img_array = np.array(image)
    
    # Sacar RGB
    R = img_array[..., 0]
    G = img_array[..., 1]
    B = img_array[..., 2]
    
    return (R, G, B)


def flatten_channel(channel):
    """
    Flattens an image to one channel
    
    :param channel:     Channel to flatten
    :return:            Flatten array
    """
    return channel.flatten()


def generate_array_image(R, G, B, height, width):
    """
    Generate the original image
    
    :param R:       The Red Channel
    :param G:       The Green Channel
    :param B:       The Blue Channel
    :param height:  The height of the image
    :param width:   The width of the image
    :return:        The image
    """
    R = R.reshape((height, width))
    G = G.reshape((height, width))
    B = B.reshape((height, width))
    
    return np.moveaxis(np.array([R, G, B]), 0, -1)
