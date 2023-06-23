# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:28:23 2023

@author: Nghia
"""

from PIL import Image
import cv2
import numpy as np

from group_kernel import IMAGE_TEXT_2x2, IMAGE_TEXT_3x2, IMAGE_TEXT_4x2

BW_THRESHOLD = 100

def thresholding(image, threshold=BW_THRESHOLD):
    image_bw = np.array(image) > threshold
    return np.uint8(image_bw)

def np_image_to_text(image, group_kernel, gh, gw):
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
    

if __name__ == "__main__":
    
    video_file = 'sample.mp4'
    cap = cv2.VideoCapture(video_file)
    
    video_text_data = {}

    success, frame = cap.read()
    c = 0
    while success:
        print(c)
        image = Image.fromarray(frame).convert("L")
        video_text_data[c] = np_image_to_text(image, IMAGE_TEXT_4x2, 4, 2)
        
        success, frame = cap.read()
        c += 1
        # assert 0
    
    for ci in range(c):
        print(video_text_data[ci])