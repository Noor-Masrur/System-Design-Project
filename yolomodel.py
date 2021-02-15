
import cv2
import numpy as np



class yoloModel:
    def __init__( self ):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.min_contour_width = 40  # 40
        self.min_contour_height = 40  # 40
        self.offset = 10  # 10
        self.line_height = 550  # 550
        self.matches = []
        self.cars = 0
        self.fileName=""

    def refreshYolo(self):
        self.min_contour_width = 40  # 40
        self.min_contour_height = 40  # 40
        self.offset = 10  # 10
        self.line_height = 550  # 550
        self.matches = []
        self.cars = 0

    def runYolo( self, fileName ):

        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(fileName)
        # cap = cv2.VideoCapture('Relaxing highway traffic.mp4')


        cap.set(3, 1920)
        cap.set(4, 1080)

        if cap.isOpened():
            ret, frame1 = cap.read()
        else:
            ret = False
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        while ret:
            d = cv2.absdiff(frame1, frame2)
            grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(grey,(5,5),0)
            blur = cv2.GaussianBlur(grey, (5, 5), 0)
            # ret , th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
            ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(th, np.ones((3, 3)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

            # Fill any small holes
            closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for (i, c) in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(c)
                contour_valid = (w >= self.min_contour_width) and (
                        h >= self.min_contour_height)

                if not contour_valid:
                    continue
                cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

                cv2.line(frame1, (0, self.line_height), (1200, self.line_height), (0, 255, 0), 2)


                #centroid = get_centroid(x, y, w, h)
                x1 = int(w / 2)
                y1 = int(h / 2)

                cx = x + x1
                cy = y + y1
                centroid=cx,cy
                self.matches.append(centroid)
                cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
                #cx, cy = get_centroid(x, y, w, h)
                for (x, y) in self.matches:
                    if y < (self.line_height + self.offset) and y > (self.line_height - self.offset):
                        self.cars = self.cars + 1
                        self.matches.remove((x, y))
                        print(self.cars)

            cv2.putText(frame1, "Total Cars Detected: " + str(self.cars), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 170, 0), 2)

            cv2.putText(frame1, "MechatronicsLAB.net", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 170, 0), 2)

            # cv2.drawContours(frame1,contours,-1,(0,0,255),2)

            cv2.imshow("Original", frame1)
            cv2.imshow("Difference", th)
            if cv2.waitKey(1) == 27:
                break
            frame1 = frame2
            ret, frame2 = cap.read()
        # print(matches)
        cv2.destroyAllWindows()
        cap.release()
    def getCarCount(self):
        return str(self.cars)
