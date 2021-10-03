# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:57:15 2021

@author: kaust
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

net = cv.dnn.readNetFromTensorflow('frozen_inference_graph_coco.pb', 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')

img = cv.imread('C:/Users/kaust/Desktop/Deep_learning/OpenCV/Image Segmentation/source code/horse.jpg')

height, width, _ = img.shape

colors = np.random.randint(0,255,(80, 3))


black = np.zeros((height, width, 3), dtype='uint8')
black[:] = (100,100,0) 

blob = cv.dnn.blobFromImage(img, swapRB=True) # Converting the image to blob format

net.setInput(blob)

boxes, masks = net.forward(["detection_out_final", "detection_masks"])# Acessing the last layers where we have the masks and boxes

total_count = boxes.shape[2] # Gives the total number of objects present in the image

print(total_count)

for i in range(total_count):

    box = boxes[0,0,i] 
    class_id = box[1]
    score = box[2]
    
    if score < 0.4:
        continue
    
    # The indexes provides the coordinate of a one object detected in a picture 
    x = int(box[3] * width) 
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)
    
    cv.rectangle(img, (x,y), (x2,y2), (0,255,0), thickness=2)
    
    roi = black[y:y2, x:x2]
    roi_height, roi_width, _ = roi.shape
    mask = masks[i, int(class_id)]
    mask = cv.resize(mask, (roi_width, roi_height))
    _, mask = cv.threshold(mask, 0.5, 255, cv.THRESH_BINARY)
    
    contours, _ = cv.findContours(np.array(mask, np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        color = colors[int(class_id)]
        cv.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
    


cv.imshow('road', img)
cv.imshow('black', black)

cv.waitKey(0) # Wait for a key to be pressed
cv.destroyAllWindows()