#!/usr/bin/env python3

# Authors: 
#	Eva Denisse Vargas 	  A01377098
# 	Brenda Vega Mendez 	  A01378360 

# Activity:Detection signal
# Date: June 16th, 2022
# For: Implementation of intelligent robotics

#Import of libraries
import rospy #mainly used in ros to initialize the node
import cv2  # used for segmentation and/or image operations
import numpy as np # Used with opencv
from sensor_msgs.msg import Image # Used to read the image from the camera
from std_msgs.msg import String, Float32
#import imutils

# Import the necessary packages:
from code import interact

#import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
import os	
from tensorflow.config import set_visible_devices
import rospkg
from statistics import mode

class detection:
    def __init__(self):

        backend.clear_session()
        set_visible_devices([], 'GPU')
        rospy.init_node("detection")
        self.rate = rospy.Rate(20)
        #-------------------------------- Subscribers-------------------------------- 
        self.image_sub = rospy.Subscriber("/video_source/raw", Image, self.image_callback)
        # -------------------------------- Publishers-------------------------------- 
        self.no_class_pub = rospy.Publisher("/no_class", String, queue_size=1)
        # -------------------------------- Variables -------------------------------- 
        self.image = None
        # -------------------------------- definition of the package to use------------------------
        RP = rospkg.RosPack()
        # -------------------------------- Training Result Path: Model------------------------------
        mainPath = os.path.join(RP.get_path("vision_codes"),"src")
        self.path_model = os.path.join(mainPath, "model")
        self.model = load_model(self.path_model)
        self.imageSize = (35, 35)
        self.str_no_class = " "
        self.prob = None
        # ----------------------------------Color parameters in HVS-------------------------------
        self.azulBajo1 = np.array([94, 112, 106], np.uint8) 
        self.azulAlto1 = np.array([137, 255, 151], np.uint8)
        self.rojoBajo1 = np.array([149,85, 34], np.uint8)
        self.rojoAlto1 = np.array([192, 255, 255], np.uint8)
	
    # ----------------Ros doesn't have cv_bridge in python3, so we use a 
    # transform to read the image. In this case, we use np.frombuffer ----------------
    def imgmsg_to_cv2(self, msg):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
   
    #---------------- Get the information from the buffer ----------------
    def image_callback(self, img):
        self.image = self.imgmsg_to_cv2(img)

    # ----------------Function to crop the image with specific information----------------
    def contours(self):
        object = False
        height, width, _ = self.image.shape
        copy = self.image.copy()
        # ------------ Blur the image ----------------
        blurred = cv2.GaussianBlur(self.image, (11, 11), 0) 
        # ---------------- Convert BGR to HSV----------------
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) 
        # ----------------Creation of the mask----------------
        mask_b = cv2.inRange(hsv, self.azulBajo1, self.azulAlto1)
        mask_r = cv2.inRange(hsv, self.rojoBajo1, self.rojoAlto1)
        mask = cv2.add(mask_r, mask_b)
        # ---------------- Erode and dilate mask ----------------
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # ----------------Find the contour of the object ----------------
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            #----------------Area to identify the object ----------------
            if (area>645) and (area<2000):
                object = True
                # ----------------used to fin the polygon that represent the STOP ----------------
                approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
                #print(approx)
                if(len(approx) == 7): 
                    #---------------- we draw the contour ----------------
                    cv2.drawContours(copy, [approx], 0, (0, 0, 255), 5) 
                # ----------------Get the center of the object ----------------
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                #---------------- we draw the circle ----------------
                cv2.circle(copy, (cx, cy), 5, (255, 0, 255), -1) 
                #---------------- we draw the contour ----------------
                cv2.drawContours(copy, contours, -1, (0, 255, 0), 2, cv2.LINE_AA) 
                #cv2.circle(img_copy, (cx, cy), 7, (255, 255, 255), 1)
                #cv2.putText(frame, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
                #---------------- used to crop the image in a rectangle ----------------
                r = int(np.sqrt(area/np.pi))
                r += 10
                x = max(cx-r, 0)
                y = max(cy-r, 0)
                w = min(cx+r, width)
                h = min(cy+r, height)
                #cv2.circle(copy, (cx, cy), 5, (255, 0, 255), -1)
                cv2.rectangle(copy, (cx-r, cy-r),(cx+r, cy+r), (0,0, 255), 1, 8, 0)
                #---------------- Final rectangle  ----------------
                roi = self.image[y:h, x:w] 
                #cv2.imshow("roi", roi)
            #cv2.imshow("Text", copy)
        return object, roi

    #---------------- Classify the image  ----------------
    def classification(self, classNo):
        if classNo == 0: return 'Ahead only'
        elif classNo == 1: return 'Stop'
        elif classNo == 2: return 'Turn right ahead'


    def main(self):
        threshold = .8
        while not rospy.is_shutdown():
            self.rate.sleep()
            if self.image is not None:
                object, roi = self.contours()
                if object:
                    image = cv2.resize(roi, self.imageSize , interpolation = cv2.INTER_AREA)
                    image = image.reshape(1, 35, 35, 3)
                    object = False 
                     #---------------- Use the model to predict the umage ----------------
                    prediction = self.model.predict(image)
                    classIn = np.argmax(prediction)
                    probValue = np.amax(prediction)
                    if (probValue>threshold):
                        str_no_class = self.classification(classIn)
                        #cv2.imshow("Prueba", copy)
                        self.no_class_pub.publish(str_no_class)
                        print(str_no_class, probValue)


if __name__ == '__main__':
    try:
        classificaation = detection()
        classificaation.main()

    except (rospy.ROSInterruptException, rospy.ROSException("topic was closed during publish()")):
        pass
