import cv2
from window import WindowManager
from camera import CaptureManager
import filters

class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager("Cameo", self.onKeypress)
        self._CaptureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)
        #self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        " 进行主循环"
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._CaptureManager.enterFrame()
            frame = self._CaptureManager.frame

            # 过滤帧
            # filters.strokeEdges(frame, frame)
            #self._curveFilter.apply(frame, frame)

            self._CaptureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """ 对特殊的按键 进行相应"""
        if keycode == 32: #空格
            self._CaptureManager.writeImage("screenhot.png")
        elif keycode == 9: # tab
            if not self._CaptureManager.isWritingVideo:
                self._CaptureManager.startWritingVideo("screencast.avi")
            else:
                self._CaptureManager.stopWritingVideo()
        elif keycode == 27:
            self._windowManager.destroyWindow()

if __name__ =="__main__":
    c1 =  Cameo()
    c1.run()
    