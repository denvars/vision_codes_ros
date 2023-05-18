#!/usr/bin/env python

# Authors: 
#	Eva Denisse Vargas 	  A01377098

# Activity: traffic light color detector
# Date: June 16th, 2022
# For: Implementation of intelligent robotics

#Import of libraries
import rospy # mainly used in ros to initialize the node
import cv2 as cv # used for segmentation and/or image operations
import numpy as np # Used with opencv
from sensor_msgs.msg import Image # Used to read the image from the camera
from cv_bridge import CvBridge # Bridge between ros and opencv
from std_msgs.msg import String 

class detection:
    def __init__(self):
        rospy.init_node("detection")

        self.rate = rospy.Rate(20)
        #self.image_pub = rospy.Publisher("/camera_topic", Image, queue_size=1)
        self.bridge = CvBridge()
        self.cv_image = None # variable used to save  the image received by the sensor
        #-------Node(s) Subscription(s)-----------
        self.image_sub = rospy.Subscriber("/video_source/raw", Image, self.image_callback)
        #-------Publisher Node(s)-----------------
        self.img_pub = rospy.Publisher("/video_source/raw/detected_img", Image, queue_size=1)
        #self.move_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.color_pub = rospy.Publisher("/color_detection", String, queue_size=1)

        # ----------------------------------Color parameters in HSV-------------------------------
        # ---------------- Green color values
        self.greenLow = np.array([60,20,85], np.uint8)
        self.greenHigh = np.array([84,255,255], np.uint8)
        self.greenMask = None
        # ---------------- Yellow color values
        self.yellowLow = np.array([16,60,125], np.uint8)
        self.yellowHigh = np.array([32,255,255], np.uint8)
        self.yellowMask = None
        #----------------  Red color values 
        self.redLow1 = np.array([0,50,130], np.uint8)
        self.redHigh1 = np.array([12,255,255], np.uint8)
        self.redLow2 = np.array([170,50,130], np.uint8)
        self.redHigh2 = np.array([180,255,255], np.uint8)
        self.redMask = None
        #
        self.dilatar = None
        # ---------------------- Parameters for the detection of a circle --------------------
        self.params = cv.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.8
        self.params.maxCircularity = 1.0
        self.params.filterByConvexity = False
        self.params.filterByInertia = False
        self.params.filterByColor = False
        self.detector = cv.SimpleBlobDetector_create(self.params)
        #self.keypoints = None
        # 
        self.redkeyp = None
        self.yellowkeyp = None
        self.greenkeyp = None
        # ----------------  Store the value detected ---------------- 
	    self.color = String() 
        #
        #self.robot_cmd = Twist()


    def image_callback(self, img):
        self.cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')

    #---------------- Red Mask Creation ---------------- 
    def redMask_detection(self, img):
        maskRed1 = cv.inRange(img, self.redLow1, self.redHigh1)
        maskRed2 = cv.inRange(img, self.redLow2, self.redHigh2)
        self.redMask = cv.add(maskRed1, maskRed2)
        color_result = cv.bitwise_and(self.cv_image, self.cv_image, mask=self.redMask)
        dilatar = self.im_processing(color_result)
        self.redkeyp = self.detector.detect(dilatar)
        #color = "red"

        if len(self.redkeyp) > 1:
            self.redColor = True
            print("The color that I am detecting is: RED")
            self.color = "red"

    #---------------- Yellow Mask Creation ---------------- 
    def yellowMask_detection(self, img):
        self.yellowMask = cv.inRange(img, self.yellowLow, self.yellowHigh)
        color_result = cv.bitwise_and(self.cv_image, self.cv_image, mask=self.yellowMask)
        dilatar = self.im_processing(color_result)
        self.yellowkeyp = self.detector.detect(dilatar)
        #color = "yellow"

        if len(self.yellowkeyp) > 1:
            self.yellowColor = True
            print("The color that I am detecting is: YELLOW")
            self.color = "yellow"

    #---------------- Green Mask Creation ---------------- 
    def greenMask_detection(self, img):
        self.greenMask = cv.inRange(img, self.greenLow, self.greenHigh)
        color_result = cv.bitwise_and(self.cv_image, self.cv_image, mask=self.greenMask)
        dilatar = self.im_processing(color_result)
        self.greenkeyp = self.detector.detect(dilatar)
        
        if len(self.greenkeyp) > 1:
            self.greenColor = True
            print("The color that I am detecting is: GREEN")	
	        self.color = "green"

    #----------------  Image proccesing ---------------- 
    def im_processing(self, mask):
        __, th1 = cv.threshold(mask, 100,255, cv.THRESH_BINARY)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
        dilatar = cv.dilate(th1, kernel, iterations=1)

        return dilatar

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            if self.cv_image is not None:
                #----------------  Image color transformation to hsv---------------- 
                frameHVS = cv.cvtColor(self.cv_image, cv.COLOR_BGR2HSV)
                # ---------------- Mask color creation---------------- 
                self.redMask_detection(frameHVS)
                self.greenMask_detection(frameHVS)
                self.yellowMask_detection(frameHVS)
                # ---------------- Final mask ---------------- 
                final_mask = self.redMask + self.yellowMask + self.greenMask
                mask =  cv.bitwise_and(self.cv_image, self.cv_image, mask = final_mask)
                final_keyp = self.redkeyp + self.yellowkeyp + self.greenkeyp
                dilatar = self.im_processing(mask)
                maskVis = cv.drawKeypoints(dilatar, final_keyp, np.array([]), (255, 0, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		        #---------------- Publishers ---------------- 
                self.img_pub.publish(self.bridge.cv2_to_imgmsg(maskVis, encoding = "bgr8"))
                self.color_pub.publish(self.color)

        cv.destroyAllWindows()

if __name__ == '__main__':
    try:
        image_processing = detection()
        image_processing.main()

    except (rospy.ROSInterruptException, rospy.ROSException("topic was closed during publish()")):
        pass
