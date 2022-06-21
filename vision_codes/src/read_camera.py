#!/usr/bin/env python
import rospy
import cv2
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class image_converter:
 
    def __init__(self):
		rospy.init_node("image_processing")
		self.rate = rospy.Rate(20)
		self.image_pub = rospy.Publisher("/camera_topic", Image, queue_size=1)
		self.bridge = CvBridge()
		self.cv_image = None
		self.image_sub = rospy.Subscriber("/video_source/raw", Image, self.image_callback)
		self.azulBajo1 = np.array([94, 112, 106], np.uint8) 
		self.azulAlto1 = np.array([137, 255, 151], np.uint8)
		self.rojoBajo1 = np.array([149,85, 34], np.uint8)
		self.rojoAlto1 = np.array([192, 255, 255], np.uint8)
		self.rojoBajo2 = np.array([127, 71, 108], np.uint8) 
		self.rojoAlto2 = np.array([255, 230, 194], np.uint8) 
 
    def image_callback(self, img):
       		self.cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')

    def main(self):
		kernel = np.ones((1,1), np.uint8) 
		while not rospy.is_shutdown():
			self.rate.sleep()
			if self.cv_image is not None:
				frameHSV = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
				maskAzul = cv2.inRange(frameHSV, self.azulBajo1, self.azulAlto1)
				maskRed1 = cv2.inRange(frameHSV, self.rojoBajo1, self.rojoAlto1)
				maskRed2 = cv2.inRange(frameHSV, self.rojoBajo2, self.rojoAlto2)
				complete_rojo = cv2.add(maskRed1, maskRed2)
				mask = cv2.add(maskAzul, complete_rojo)
				img_dilation = cv2.dilate(mask.copy(), kernel, iterations=1) 
				output_image = self.bridge.cv2_to_imgmsg(img_dilation, encoding = 'passthrough')	
				self.image_pub.publish(output_image)
	
    	     	cv2.destroyAllWindows()

if __name__ == '__main__':
   	try:
		image_processing = image_converter()
		image_processing.main()

	except (rospy.ROSInterruptException, rospy.ROSException("topic was closed during publish()")):
		pass
