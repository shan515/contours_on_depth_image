#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import roslib
roslib.load_manifest('begineer_tutorial')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

contours = np.array([0])
#depth = np.array([0])
img = np.array([])
res = np.array([])

class image_converter:

  def __init__(self):
    
    self.bridge = CvBridge()
    #self.depth_sub = rospy.Subscriber("/camera/depth/image",Image,self.callback)
    self.image_sub = rospy.Subscriber("/camera/rgb/image_color",Image,self.callagain)
    self.depth_sub = rospy.Subscriber("/camera/depth/image",Image,self.callback)


  def callagain(self,data):
    global contours
    global res
    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    image = cv2.cvtColor(cv_image , cv2.COLOR_BGR2HSV)
    lower_range = np.array([30,150,50])
    upper_range = np.array([255,255,180])
    mask = cv2.inRange(image , lower_range, upper_range)
    res = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

  def callback(self,data):
    global contours
    global res
    depth = self.bridge.imgmsg_to_cv2(data, "32FC1")
    #cnt = contours[0]
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(depth,(x,y),(x+w,y+h),(0,255,0),2)
    #img = cv2.drawContours(depth, contours, -1, (255,255,255), 3)
    cv2.imshow("Depth window", depth)
    cv2.imshow("Image window", res) 
    cv2.waitKey(5)
    #print("pixel:" , img[240][320])
    #print("contours", cnt)

    
  
def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
      rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    







      

