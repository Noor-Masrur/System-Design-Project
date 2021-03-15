import cv2
import os
import numpy as np
from PIL.ImageQt import ImageQt

from model import Model
from PIL import Image as im

class regionOfInterest():
    def __init__(self):
        #super.__init__(self)
        #self.mdl=Model()
        #fileName=self.mdl.getFileName(  )
    #    self.cap2 = cv2.VideoCapture(fileName)
        # cap = cv2.VideoCapture('Relaxing highway traffic.mp4')
    #    self.cap2.set(3, 1920)
    #    self.cap2.set(4, 1080)
    #    self.frame12=None
    #    self.frame22=None
    #    self.cap=None
    #    if self.cap2.isOpened():
    #        self.ret2, self.frame12 = self.cap2.read()
    #    else:
    #        self.ret2 = False
    #    self.ret2, self.frame12 = self.cap2.read()
    #    self.ret2, self.frame22 = self.cap2.read()

        self.original_image = cv2.imread('preview.jpg')
        self.clone = self.original_image.copy()
        self.count=0
        cv2.namedWindow('Put Region Of Interest')
        self.image_coordinates = []
        cv2.setMouseCallback('Put Region Of Interest', self.extract_coordinates)

        # List to store start/end points


    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click

        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            print(self.image_coordinates[self.count])
            self.count+=1


                # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            print(self.image_coordinates[self.count])
            self.count+=1
            #print(self.image_coordinates[cnt-1])
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("Put Region Of Interest", self.clone)


    def get_points(self):
        return self.image_coordinates

                # Draw line

            # Clear drawing boxes on right mouse button click
            #elif event == cv2.EVENT_RBUTTONDOWN:
                #self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone
