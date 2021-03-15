from numba import jit, cuda
import cv2
import numpy as np
import sys
from RegionOfInterest import regionOfInterest
import time
from PIL import Image as im
from PIL.ImageQt import ImageQt
from PIL import Image, ImageTk
import temp as od
import imageio
import argparse

class yoloModel():
    def __init__( self ):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        super().__init__()
        self.pos = []
        self.line = []
        self.counter = 0
        self.w, self.h = (1366, 768)
        self.reader=None
        self.writer=None
        self.fps=None
        self.flag=True
        self.boxes=[]
        self.image=None
        self.classWiseCount=None

        self.cars = 0

        self.original=None
        self.difference=None
        self.cap=None
        self.frame1=None
        self.frame2=None
        self.ret= None
        self.th=None
        self.region=None
        self.j=0

    def refreshYolo(self):
        self.pos = []
        self.line = []
        self.rect = []
        self.counter = 0
        self.w, self.h = (1366, 768)
        self.reader=None
        self.writer=None
        self.fps=None
        self.flag=True
        self.boxes=[]
        self.image=None
        self.j=0
        self.classWiseCount=None

        self.cars = 0
        self.original=None
        self.difference=None
        self.cap=None
        self.frame1=None
        self.frame2=None
        self.ret= None
        self.th=None
        self.region=None

    #@jit(target ="cuda")
    def intersection(self, p, q, r, t):
        #print(p, q, r, t)
        (x1, y1) = p
        (x2, y2) = q

        (x3, y3) = r
        (x4, y4) = t

        a1 = y1-y2
        b1 = x2-x1
        c1 = x1*y2-x2*y1

        a2 = y3-y4
        b2 = x4-x3
        c2 = x3*y4-x4*y3

        if(a1*b2-a2*b1 == 0):
            return False
        #print((a1, b1, c1), (a2, b2, c2))
        x = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
        y = (a2*c1 - a1*c2) / (a1*b2 - a2*b1)
        #print((x, y))

        if x1 > x2:
            tmp = x1
            x1 = x2
            x2 = tmp
        if y1 > y2:
            tmp = y1
            y1 = y2
            y2 = tmp
        if x3 > x4:
            tmp = x3
            x3 = x4
            x4 = tmp
        if y3 > y4:
            tmp = y3
            y3 = y4
            y4 = tmp

        if x >= x1 and x <= x2 and y >= y1 and y <= y2 and x >= x3 and x <= x4 and y >= y3 and y <= y4:
            return True
        else:
            return False

    def runYolo( self, fileName ):


        self.cap = cv2.VideoCapture(fileName)
        #self.reader = imageio.get_reader(fileName)
        #self.fps = self.reader.get_meta_data()['fps']
        #self.writer = imageio.get_writer(r'C:\Users\Asus\Desktop\tausif\output.mp4', fps = fps)
        self.ret, self.frame1 = self.cap.read()
        self.ret, self.frame2 = self.cap.read()
        #cv2.imwrite('tt.jpg', self.frame1)
        #chobi=image
        #self.ret, self.frame1 = self.cap.read()
        #self.ret, self.frame2 = self.cap.read()
        chobi=im.fromarray(self.frame1,'RGB')
        chobi.save('preview.jpg')
        self.get_region=regionOfInterest()
        while True:
            cv2.imshow('Put Region Of Interest', self.get_region.show_image())
            key = cv2.waitKey(1)
                # Close program with keyboard 'q'
            if key == ord('q'):
                self.line=self.get_region.get_points()
                cv2.destroyAllWindows()
                break;

                #exit(1)
        #time.sleep(60)
        self.j=1
        return self.ret

    #@jit(target ="cuda")
    def loop (self) :

        self.ret, self.frame1 = self.cap.read()
        self.ret, self.frame2 = self.cap.read()

        if (type(self.frame1) == type(None)):
            #self.writer.close()
            self.flag=False

        image_h, image_w, _ = self.frame1.shape
        new_image = od.preprocess_input(self.frame1, od.net_h, od.net_w)

            # run the prediction
        yolos = od.yolov3.predict(new_image)
        self.boxes = []

        for i in range(len(yolos)):
                # decode the output of the network
            self.boxes += od.decode_netout(yolos[i][0], od.anchors[i], od.obj_thresh, od.nms_thresh, od.net_h, od.net_w)

            # correct the sizes of the bounding boxes
        od.correct_yolo_boxes(self.boxes, image_h, image_w, od.net_h, od.net_w)

            # suppress non-maximal boxes
        od.do_nms(self.boxes, od.nms_thresh)

            # draw bounding boxes on the image using labels
        self.image = od.draw_boxes(self.frame1, self.boxes, self.line, od.labels, od.obj_thresh, self.j)
        #writer.append_data(self.image)
        d = cv2.absdiff(self.frame1, self.frame2)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        self.ret, self.th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        
        return self.image

            # cv2.imwrite('E:/Virtual Traffic Light Violation Detection System/Images/frame'+str(j)+'.jpg', image2)
            # self.show_image('E:/Virtual Traffic Light Violation Detection System/Images/frame'+str(j)+'.jpg')

        #cv2.imshow('Traffic Violation', image2)

        #print(j)


    def loop2 (self):
        self.j+=1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            #writer.close()
            self.flag=False
            return False
        self.frame1 = self.frame2
        self.ret, self.frame2 = self.cap.read()
        return True
        # print(matches)
        #j = j+1

    def getClassWiseCount(self):
        self.classWiseCount=od.get_vehicle_count()
        od.refresh_vehicle_count()
        return self.classWiseCount






    def getCarCount(self):
        return str(self.cars)
