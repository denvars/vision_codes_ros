# File        :   classify.py (Pokenet's testing script)
# Version     :   1.0.2
# Description :   Script that calls the Pokenet CNN and tests it on example images.
# Date:       :   May 04, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

# Import the necessary packages:
from code import interact
from configparser import Interpolation
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
from imutils import paths
import imutils
import os

videoCap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.5
isObject = False
current = ""

# Reads image via OpenCV:
def readImage(imagePath):
    # Open image:
    print("readImage>> Reading: " + imagePath)
    inputImage = cv2.imread(imagePath)
    # showImage("Input Image", inputImage)

    if inputImage is None:
        print("readImage>> Could not load Input image.")

    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)

def classification(classNo):
    if classNo == 0: return 'Ahead only'
    elif classNo == 1: return 'Stop'
    elif classNo == 2: return 'Turn right'
   
 

# Set the resources paths:
# mainPath = "D://CNN//pokenet//"
mainPath = os.path.join("/home/edevars", "cnn-edevars")
# modelPath = mainPath + "output//"
modelPath = os.path.join(mainPath, "output_signals")
#print(modelPath)
# Training image size:
imageSize = (35, 35)

# Load model:
path_model = os.path.join(modelPath, "model")
model = load_model(path_model)
#lb = pickle.loads(open(os.path.join(modelPath, "labels.pickle"), "rb").read())

azulBajo1 = np.array([61, 59, 115], np.uint8) 
azulAlto1 = np.array([136, 255, 255], np.uint8) 

rojoBajo1 = np.array([107, 95, 123], np.uint8) 
rojoAlto1 = np.array([190, 255, 255], np.uint8)

#rojoBajo2 = np.array([0, 95, 152], np.uint8) 
#rojoAlto2 = np.array([255, 131, 255], np.uint8)  
list_signal = []
center = None
object = False
#negroBajo = np.array([96, 2, 0], np.uint8) 
#negroAlto = np.array([162, 72, 163], np.uint8) 

#def detection_mask():

while True: 
    ret, frame = videoCap.read()
    height, width, _ = frame.shape
    copy = frame.copy()
    kernel=np.ones((5,5), np.uint8)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_b = cv2.inRange(hsv, azulBajo1, azulAlto1)
    mask_r = cv2.inRange(hsv, rojoBajo1, rojoAlto1)
    #mask_r1 = cv2.inRange(hsv, rojoBajo2, rojoAlto2)
    #maskr = cv2.add(mask_r, mask_r1)
    mask = cv2.add(mask_r, mask_b)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        #rect = cv2.minAreaRect(area)
        if (area>5000) and (area<7500):
            object = True
            approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
            #print(approx)
            if(len(approx) == 7): 
                cv2.drawContours(copy, [approx], 0, (0, 0, 255), 5)
            object = True
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(copy, (cx, cy), 5, (255, 0, 255), -1)
            cv2.drawContours(copy, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
            #print( cv2.circle(copy, (cx, cy), 5, (255, 0, 255), -1))
            #cv2.circle(img_copy, (cx, cy), 7, (255, 255, 255), 1)
            #cv2.putText(frame, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
            r = int(np.sqrt(area/np.pi))
            r += 10
            x = max(cx-r, 0)
            y = max(cy-r, 0)
            w = min(cx+r, width)
            h = min(cy+r, height)
            #cv2.circle(copy, (cx, cy), 5, (255, 0, 255), -1)
            cv2.rectangle(copy, (cx-r, cy-r),(cx+r, cy+r), (0,0, 255), 1, 8, 0)
            roi = frame[y:h, x:w]
            cv2.imshow("roi", roi)
        #cv2.imshow("Text", copy)
        if object:
            isObject = False 
            height_r, width_r, _ = roi.shape
            #image = cv2.medianBlur(roi, 5)
            image = cv2.resize(roi, (32, 32), interpolation = cv2.INTER_AREA)
            image = image.reshape(1, 32, 32, 3)
            cv2.putText(frame, "CLASS: " , (20, 35), font, 0.75, (128, 0, 128), 2, cv2.LINE_AA)
            cv2.putText(frame, "PROBABILITY: ", (20, 75), font, 0.75, (128, 0, 128), 2, cv2.LINE_AA)
            prediction = model.predict(image)
            classIn = np.argmax(prediction)
            probValue = np.amax(prediction)
            #last = str_no_class
            if (probValue>threshold):
                str_no_class = classification(classIn)
                    #list_signal.append(str_no_class)
                    #print(list_signal[-1])
                #cv2.imshow("Prueba", copy)
                print(str_no_class, probValue)
                #last = str_no_class
            else:
                    print("Sin señal")
            cv2.imshow("Prueba", copy)


    if cv2.waitKey(1) and 0xFF == ord('q'):
       break
