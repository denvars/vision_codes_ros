#!/usr/bin/env python

# Authors: 
#	Eva Denisse Vargas 	  A01377098
# 	Brenda Vega Mendez 	  A01378360 

# Activity: color slider 
# Date: June 16th, 2022
# For: Implementation of intelligent robotics

import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
 
def nothing(x):
   pass
 
#We create a window called 'image' in which there will be all the sliders
cv2.namedWindow('image')
cv2.createTrackbar('Hue Minimo','image',0,255,nothing)
cv2.createTrackbar('Hue Maximo','image',0,255,nothing)
cv2.createTrackbar('Saturation Minimo','image',0,255,nothing)
cv2.createTrackbar('Saturation Maximo','image',0,255,nothing)
cv2.createTrackbar('Value Minimo','image',0,255,nothing)
cv2.createTrackbar('Value Maximo','image',0,255,nothing)
 
while True:
  _,frame = cap.read() 
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Convert it to HSV color space
 
  #The maximum and minimum values ​​of H,S and V are saved based on the position of the sliders
  hMin = cv2.getTrackbarPos('Hue Minimo','image')
  hMax = cv2.getTrackbarPos('Hue Maximo','image')
  sMin = cv2.getTrackbarPos('Saturation Minimo','image')
  sMax = cv2.getTrackbarPos('Saturation Maximo','image')
  vMin = cv2.getTrackbarPos('Value Minimo','image')
  vMax = cv2.getTrackbarPos('Value Maximo','image')
 
  #An array is created with the minimum and maximum positions
  lower=np.array([hMin,sMin,vMin])
  upper=np.array([hMax,sMax,vMax])
 
  #color detection
  mask = cv2.inRange(hsv, lower, upper)
 
  #Show results and exit
  cv2.imshow('camara',frame)
  cv2.imshow('mask',mask)
  k = cv2.waitKey(5) & 0xFF
  if k == 27:
    break
cv2.destroyAllWindows()
