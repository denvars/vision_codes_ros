#!/usr/bin/env python
import rospy 
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

class movement: 
	def __init__(self):
		self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
		self.wl_subs = rospy.Subscriber("/wl", Float32, self.wlCallback)
		self.wr_subs =rospy.Subscriber("/wr", Float32, self.wrCallback)
		self.robot_cmd = Twist()
		self.rate = rospy.Rate(5)
		self.flag_stop_move = False 
		self.flag_stop_rotate = False
		self.R, self.L, self.robot_wl, self.robot_wr =  0.055, 0.185, 0.0, 0.0 
		rospy.on_shutdown(self.endCallback)

	def wlCallback(self, msg):
		self.robot_wl = msg.data

	def wrCallback(self,msg): 
		self.robot_wr = msg.data

	def endCallback(self):
		self.robot_cmd.linear.x = 0.0
		self.robot_cmd.angular.z = 0.0
		self.pub.publish(self.robot_cmd)

	def move(self, fwd_speed=0.0, dist=0.0):
		self.robot_cmd.linear.x = fwd_speed
		self.robot_cmd.angular.z = 0.0 
		estimated_dist = 0.0
		t0 = rospy.get_rostime().to_sec()
		while(estimated_dist <= dist): #and self.flag_stop_move == False:
			self.pub.publish(self.robot_cmd)
			t1 = rospy.get_rostime().to_sec()
			estimated_dist += self.R*((self.robot_wr+self.robot_wl)/2)*(t1- t0)
			t0 = t1
			self.rate.sleep()
		self.robot_cmd.linear.x = 0.0
		self.pub.publish(self.robot_cmd)
		while(abs(self.robot_wl)> 0.01 or abs(self.robot_wr)>0.001):
			self.rate.sleep()
		

	def rotate(self, angular_speed=0.0, angle=0.0):
		self.robot_cmd.linear.x = 0.0
		self.robot_cmd.angular.z = angular_speed
		estimated_angle = 0.0
		t0 = rospy.get_rostime().to_sec()
		while(abs(estimated_angle) < abs(angle)):
			self.pub.publish(self.robot_cmd)
			t1 = rospy.get_rostime().to_sec()
			estimated_angle += self.R*(self.robot_wr-self.robot_wl)*(t1 - t0)/self.L
			t0 = rospy.get_rostime().to_sec()
			self.rate.sleep()
		self.robot_cmd.angular.z = 0.0
		self.pub.publish(self.robot_cmd)
		while(abs(self.robot_wl)> 0.01 or abs(self.robot_wr)>0.01):
			print("Deteniendo rotacion")
			self.rate.sleep()
