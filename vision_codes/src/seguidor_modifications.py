#!/usr/bin/env python

# Autores: 
#	Eva Denisse Vargas 	  A01377098
# 	Brenda Vega Mendez 	  A01378360 

# Actividad: Seguidor de linea 
# fecha: 19 de mayo del 2022
# Materia: Implementacion de robotica inteligente 


import rospy 
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool, Float32, String

import cv2 
import numpy as np
from math import pi
import square

class processing:
	def __init__(self):
		rospy.init_node("seguidor")
		self.rate = rospy.Rate(5)
	# Variable del topico de la camara
		self.image_sub = rospy.Subscriber("/video_source/raw", Image, self.image_callback)	
	# Variable del topico publicador de la camara
		self.img_pub = rospy.Publisher("/video_source/raw/detected_line_img", Image, queue_size=1)
		self.move_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10) 
	# Topico del objecto que detecta 
		#self.isObject_pub = rospy.Publisher("isObjectLine", Bool, queue_size=1)
		self.bridge = CvBridge()
		self.cmd_vel = Twist()
		self.color_subs = rospy.Subscriber("/color_detection", String, self.color_callback)
		self.no_class_subs = rospy.Subscriber("/no_class", String, self.str_no_class_callback)
		#self.prob_subs = rospy.Subscriber("/prob_value", Float32, self.prob_callback)
	# variable que guarda la imagen que obtiene del subscritor
		self.frame = None
		self.color = None
		self.str_no_class = None
		#self.prob = Float32()
              #self.bandera_deteccion = False
		self.move = square.movement()
              #print(str_no_class, probValue)
              
		
	# Funcion que convierte lo que ve el sensor a una imagen 
        def image_callback(self, img):
               self.frame = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        def color_callback(self, msg): 
               self.color = msg.data
        def str_no_class_callback(self, msg):
               self.str_no_class = msg.data
        def prob_callback(self, msg):
               self.prob = msg.data
    
    # Deteccion de un objeto 
        def processing_img(self, img):
               kernel = np.ones((5,5),np.uint8)
               # Aplicacion de los filtros para eliminar el ruido de la imagen
               gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
               __, binary = cv2.threshold(gray, 100,255, cv2.THRESH_BINARY_INV)
               opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
               # Publica la imagen binaria 
               self.img_pub.publish(self.bridge.cv2_to_imgmsg(binary, encoding = "passthrough"))
               return opening

   # Obtencion el contorno de la figura
        def figure_con(self, opening):
			contours,_  = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			isObject = False     # Verdadero si encuentra un objeto
			cx,cy = 0,0          #Coordenadas 
			minArea = 500  # Area minima para considerar que es un objeto
			for cnt in contours:
				momentos = cv2.moments(cnt)
				area = momentos['m00']
				if (area>minArea):
					cx = int(momentos['m10']/momentos['m00'])    
					cy = int(momentos['m01']/momentos['m00'])
					isObject = True
			return isObject,cx,cy

    # Calculo del error 
        def controller(self, isObject, cx, hxd):
                        if isObject:
                            hx = self.frame.shape[1]/2-cx # Obtiene el centro de la figura             
                            hxe  = hxd-hx # Calculo del error (diferencia entre el centro de la imagen y el centro de la figura)        
                            k = 0.002 # Variable proporcional del controlador :
                            if self.str_no_class == "Stop" or self.color =="red":
                                   v = 0.0
                                   w = 0.0
                            else: 
                                   #print("movimiento")
                                   v = 0.05 # velocidad lineal a la que se move el bot 
                                   w = -k*hxe # calculo de la velocidad angular
                        else:
                               v = 0.0
                               w = 0.0
                            
                            #print("calculo del error: ", w)
    
    	              	# Publicacion de las varibales calculadas 
                      	self.cmd_vel.linear.x = v
                      	self.cmd_vel.angular.z = w	
                      	self.move_pub.publish(self.cmd_vel)

    # Funcion principal 
    	def main(self):
            flag_detection_turnRigh = False
            while not rospy.is_shutdown():
                    self.rate.sleep()
                    if self.frame is not None:
                            # primer recorte de la imagen 
                            frame = self.frame[(self.frame.shape[0]//2):, 0:(self.frame.shape[1])]
                            # Segundo recorte de la imagen 
                            cropframe = frame[(frame.shape[0]//2):, 0:(frame.shape[1])]
                            crop_frame2 = cropframe[(cropframe.shape[0]//2):, 0:(cropframe.shape[1])]
                            # ----- Obtiene las coordenadas de la imagen recortada --------
                            cxd = int(crop_frame2.shape[1]/2)
                            cyd = int(crop_frame2.shape[0]/2)
                            # ----- Calculo del error deseado ---------
                            hxd = int(crop_frame2.shape[1]/2) - cxd
                            # Se manda a la funcion "ObjectDetection" para obtener las variables 
                            opening = self.processing_img(crop_frame2)
                            isObject, cx, cy = self.figure_con(opening)
                            # __________________   Posicion de los circulos _______________
                            cv2.circle(crop_frame2,(cx,cy),10, (0,0,255), -1) # figura 
                            cv2.circle(crop_frame2,(cxd,cyd),10, (0,255,0), -1) # Imagen
                            # Se manda a la funcion del movimiento
                            #print("Detected signal", self.str_no_class)

                            # ______________________ Deteccion de las señales _______________ 
                            if isObject == False:
                                   print("Detected signal", self.str_no_class)
                                   if self.str_no_class == "Turn right ahead" : 
                                          self.move.move(0.1, 0.32)
                                          self.move.rotate(-0.1, pi/2)
                                          self.move.move(0.1, 0.32)
                                          
                                   elif self.str_no_class == "Ahead only":
                                          self.move.move(0.1, 0.7)
                                          print("Derecho En movimiento in ", isObject )
                                   
                            else: 
                                   self.controller(isObject, cx, hxd)
                                  
                                  
                            #cv2.destroyAllWindows()

if __name__ == '__main__':
	try:
		detection = processing()
		detection.main()
	except (rospy.ROSInterruptException, rospy.ROSException("topic was closed during publish()")):
		pass
