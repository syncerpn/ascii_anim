# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:28:23 2023

@author: Nghia
"""

from PIL import Image
import numpy as np
from multiprocessing import Process, Manager, Lock
import matplotlib.pyplot as plt

BW_THRESHOLD = 160

IMAGE_TEXT_2x2 = {
    "0000": " ",
    "0001": ",",
    "0010": ".",
    "0011": "_",
    "0100": "'",
    "0101": "]",
    "0110": "/",
    "0111": "J",
    "1000": "`",
    "1001": "\\",
    "1010": "[",
    "1011": "L",
    "1100": "^",
    "1101": "?",
    "1110": "P",
    "1111": "@",
    }

IMAGE_TEXT_3x2 = {
    "000000": " ",
    "000001": ",",
    "000010": ".",
    "000011": "_",
    "000100": "-",
    "000101": ":",
    "000110": ";",
    "000111": "=",
    "001000": "-",
    "001001": ";",
    "001010": ":",
    "001011": "<",
    "001100": "-",
    "001101": ">",
    "001110": "<",
    "001111": "=",
    "010000": "'",
    "010001": "i",
    "010010": ":",
    "010011": "J",
    "010100": "!",
    "010101": "|",
    "010110": "/",
    "010111": "J",
    "011000": "'",
    "011001": "{",
    "011010": "/",
    "011011": "(",
    "011100": "\"",
    "011101": "+",
    "011110": "/",
    "011111": "d",
    "100000": "`",
    "100001": "\\",
    "100010": "!",
    "100011": "L",
    "100100": "\\",
    "100101": ")",
    "100110": "{",
    "100111": "J",
    "101000": "!",
    "101001": "(",
    "101010": "|",
    "101011": "L",
    "101100": "\"",
    "101101": "\\",
    "101110": "+",
    "101111": "b",
    "110000": "*",
    "110001": "?",
    "110010": "!",
    "110011": "=",
    "110100": "\"",
    "110101": "?",
    "110110": "?",
    "110111": "]",
    "111000": "\"",
    "111001": "(",
    "111010": "T",
    "111011": "[",
    "111100": "*",
    "111101": "?",
    "111110": "P",
    "111111": "@",
    }

def thresholding(image, threshold=BW_THRESHOLD):
    image_bw = np.array(image) > threshold
    return np.uint8(image_bw)

def np_image_to_text(image, group_kernel=IMAGE_TEXT_2x2, gh=2, gw=2):
    image_bw = thresholding(image)
    h, w = image_bw.shape
    
    image_bw = image_bw[:h//gh*gh, :w//gw*gw]
    hh, ww = image_bw.shape
    
    image_array = list(image_bw.tolist())
    hhh = hh // gh
    
    www = ww // gw
    
    image_array_reform = []
    for j in range(hhh):
        for i in range(www):
            image_array_reform_char = []
            for k in range(gh):
                image_array_reform_char += image_array[j*gh+k][i*gw:i*gw+gw]
        
            image_array_reform += [image_array_reform_char]
        
    image_text_reform = [group_kernel[''.join(list(map(str, c)))] for c in image_array_reform]
    image_text = ["".join(image_text_reform[i*www:i*www+www-1]) for i in range(hhh)]
    return "\n".join(image_text)

image = Image.open("sample.png").convert("L")
# image_text = np_image_to_text(image)
# print(image_text)

image_text = np_image_to_text(image, group_kernel=IMAGE_TEXT_3x2, gh=3, gw=2)
print(image_text)