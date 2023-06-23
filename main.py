# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:28:23 2023

@author: Nghia
"""

from PIL import Image
import numpy as np
from multiprocessing import Process, Manager, Lock
import matplotlib.pyplot as plt

from group_kernel import IMAGE_TEXT_2x2, IMAGE_TEXT_3x2, IMAGE_TEXT_4x2

BW_THRESHOLD = 160

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

image_text = np_image_to_text(image, group_kernel=IMAGE_TEXT_4x2, gh=4, gw=2)
print(image_text)