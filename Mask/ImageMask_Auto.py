#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
[Machine Learning and Programming] 2022 Spring Semester 
Project 5  
'''

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


start = False

hsv = [0, 204, 100]
lb1 = 0
ub1 = 0
lb2 = 0
ub2 = 0
lb3 = 0
ub3 = 0

if hsv[0] < 10:
    lb1 = np.array([hsv[0]-10+180, 30, 30])
    ub1 = np.array([180, 255, 255])
    lb2 = np.array([0, 30, 30])
    ub2 = np.array([hsv[0], 255, 255])
    lb3 = np.array([hsv[0], 30, 30])
    ub3 = np.array([hsv[0]+10, 255, 255])  
elif hsv[0] > 170:
    lb1 = np.array([hsv[0], 30, 30])
    ub1 = np.array([180, 255, 255])
    lb2 = np.array([0, 30, 30])
    ub2 = np.array([hsv[0]+10-180, 255, 255])
    lb3 = np.array([hsv[0]-10, 30, 30])
    ub3 = np.array([hsv[0], 255, 255])          
else:
    lb1 = np.array([hsv[0], 30, 30])
    ub1 = np.array([hsv[0]+10, 255, 255])
    lb2 = np.array([hsv[0]-10, 30, 30])
    ub2 = np.array([hsv[0], 255, 255])
    lb3 = np.array([hsv[0]-10, 30, 30])
    ub3 = np.array([hsv[0], 255, 255])

imgOrg_path = 'Mask/input_image/'
file_lst = os.listdir(imgOrg_path)

for file in file_lst:
    name = imgOrg_path + file
    sav_name = 'Mask/color_segmentation/' + file
    img_color = cv.imread(name)
    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)
        
    imgHsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
        
    imgMask1 = cv.inRange(imgHsv, lb1, ub1)
    imgMask2 = cv.inRange(imgHsv, lb2, ub2)
    imgMask3 = cv.inRange(imgHsv, lb3, ub3)
    imgMask = imgMask1 | imgMask2 | imgMask3
    img_result = cv.bitwise_and(img_color, img_color, mask=imgMask)
        
    cv.imwrite(sav_name, img_result)


# In[3]:


imgSeg_path = "Mask/color_segmentation/"
savePath = 'Mask/mask_image/'
file_lst = os.listdir(imgSeg_path)
for file in file_lst:
    filepath = imgSeg_path + file
    savName = savePath + file
    img = cv.imread(filepath, cv.IMREAD_COLOR)
    
    if img is None:
        print("fail to load the image, try again")
        sys.exit()
    
    dst = cv.medianBlur(img, 3)
    cv.imwrite(savName, dst)


# In[4]:


K = 10
imgNof_path = 'Mask/mask_image/'
file_lst2 = os.listdir(imgSeg_path)
for file in file_lst2:
    filepath = imgNof_path + file
    img = cv.imread(filepath)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    
    criteria = (cv.TERM_CRITERIA_EPS + 
                cv.TERM_CRITERIA_MAX_ITER,10,1.0)
    
    attempts = 10 
    ret,label,center = cv.kmeans(vectorized,2,None,criteria,attempts,
                                  cv.KMEANS_PP_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()] * [255,255,255] 
    result_image = res.reshape((img.shape))
    
    
    cv.imwrite(filepath, result_image)


# In[5]:


# morphological operations
for file in file_lst2:
    filepath = imgNof_path + file
    img = cv.imread(filepath, 0)
    
    kernel = np.ones((5,5), np.uint8)
    result_image = cv.dilate(img, kernel, iterations = 1)

    cv.imwrite(filepath, result_image)


# In[6]:


for file in file_lst2:
    filepath = imgNof_path + file
    #print(filepath)
    savName = savePath + file
    #img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    img = cv.imread(filepath, cv.IMREAD_COLOR)
    
    if img is None:
        print("fail to load the image, try again")
        sys.exit()
    
    dst = cv.medianBlur(img, 3)
    
    cv.imwrite(savName, dst)

