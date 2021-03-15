# MyApp.py
# D. Thiebaut
# PyQt5 Application
# Editable UI version of the MVC application.
# Inherits from the Ui_MainWindow class defined in mainwindow.py.
# Provides functionality to the 3 interactive widgets (2 push-buttons,
# and 1 line-edit).
# The class maintains a reference to the model that implements the logic
# of the app.  The model is defined in class Model, in model.py.
from PIL import Image as im
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog
from mainwindow import Ui_MainWindow
import sys
from model import Model
from yolomodel import yoloModel
import cv2
import numpy as np
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.Qt import QUrl
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap
from PIL.ImageQt import ImageQt
from PyQt5.QtGui import QImage
import os
# MyApp.py
from numba import jit, cuda
class MainWindowUIClass( Ui_MainWindow ):
    def __init__( self ):
        '''Initialize the super class
        '''
        super().__init__()
        self.model = Model()
        self.yolomodel = yoloModel()
        self.yoloNibo=False
        self.ssdNibo=False
        self.player=QMediaPlayer()
        self.result=None
        self.videoPlayer=QVideoWidget()

    def setupUi( self, MW ):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super().setupUi( MW )

        # close the lower part of the splitter to hide the
        # debug window under normal operations
        #self.splitter = QtWidgets.QSplitter()
        #self.splitter.setSizes([300, 0])

    def debugPrint( self, msg ):
        '''Print the message in the text edit at the bottom of the
        horizontal splitter.
        '''
        self.debugTextBrowser.append( msg )

    def get_original_video(self):
        return self.original_video

    def refreshAll( self ):
        '''
        Updates the widgets whenever an interaction happens.
        Typically some interaction takes place, the UI responds,
        and informs the model of the change.  Then this method
        is called, pulling from the model information that is
        updated in the GUI.
        '''
        self.lineEdit.setText( self.model.getFileName() )
        '''self.yolomodel.min_contour_width = 40  # 40
        self.yolomodel.min_contour_height = 40  # 40
        self.yolomodel.offset = 10  # 10
        self.yolomodel.line_height = 550  # 550
        self.yolomodel.matches = []
        self.yolomodel.cars = 0'''

        #self.textEdit.setText( self.model.getFileContents() )

    # slot
    def returnPressedSlot( self ):
        ''' Called when the user enters a string in the line edit and
        presses the ENTER key.
        '''
        fileName =  self.lineEdit.text()
        if self.model.isValid( fileName ):
            self.model.setFileName( self.lineEdit.text() )
            self.refreshAll()
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid file name!\n" + fileName )
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = m.exec_()
            self.lineEdit.setText( "" )
            self.refreshAll()
            self.debugPrint( "Invalid file specified: " + fileName  )

    # slot

    # slot
    def browseSlot( self ):
        ''' Called when the user presses the Browse button
        '''
        #self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        if fileName:
            self.debugPrint( "File Location : " + fileName )
            self.model.setFileName( fileName )
            self.refreshAll()
    def yoloSlot(self):
        if self.yolo.isChecked :
            #self.debugPrint( "kaj kore" )
            self.yoloNibo=True
            self.ssdNibo=False
        #return False
    def ssdSlot(self):
        if self.ssd.isChecked :
            self.yoloNibo=False
            self.ssdNibo=True
    #@jit(target ="cuda")
    def reportSlot(self):
        #self.player.setVideoOutput(self.videoPlayer)
        #self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.model.fileName)))
        #self.player.play()
        #self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.model.fileName)))
        #self.player.setMedia(QMediaContent(self.yolomodel.original))
        if self.yoloNibo:
            self.yolomodel.ret = self.yolomodel.runYolo(self.model.fileName)

            while self.yolomodel.ret and self.yolomodel.flag:

                frame=self.yolomodel.loop()
                chobi=im.fromarray(frame,'RGB')
                chobi.save('1.jpg')
                chobi = QPixmap('1.jpg').scaled(self.original_video.width(),self.original_video.height())
                self.original_video.setPixmap(chobi)
                #self.original_video.resize(pixmapp.width(),pixmapp.height())
                self.original_video.show()

                porer_chobi=im.fromarray(self.yolomodel.th)
                porer_chobi.save('2.jpg')
                porer_chobi = QPixmap('2.jpg').scaled(self.difference_video.width(),self.difference_video.height())
                self.difference_video.setPixmap(porer_chobi)
                #self.original_video.resize(pixmapp.width(),pixmapp.height())
                self.original_video.show()
                self.difference_video.show()

                flag=self.yolomodel.loop2()
                if not flag :
                    break
                self.result=self.yolomodel.getClassWiseCount()
                for i in self.result:
                    print(i +" "+str(self.result[i]))
            cv2.destroyAllWindows()
            self.yolomodel.cap.release()
            self.debugPrint(  "Final Count :")
            self.result=self.yolomodel.getClassWiseCount()
            for i in self.result:
                self.debugPrint(i +" "+str(self.result[i]))
            self.yolomodel.refreshYolo()
            os.remove('1.jpg')
            os.remove("2.jpg")




def main():
    """
    This is the MAIN ENTRY POINT of our application.  The code at the end
    of the mainwindow.py script will not be executed, since this script is now
    our main program.   We have simply copied the code from mainwindow.py here
    since it was automatically generated by '''pyuic5'''.

    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUIClass()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

main()
