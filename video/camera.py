import cv2
import numpy
import time
# fps frames per second   每秒传输的帧数

class CaptureManager(object):
    # 进入画面
    # 退出画面（绘制到窗口，写入文件， 释放帧）
    # 写入图像
    # 开始录像 （捕获摄像头的画面）
    # 停止录像 存储录像
# args:
#     capture:VideoCapture 对象， 用于读取视频文件或者 捕捉摄像头图像（capture = cv2.VideoCapture(0)）
#     preview_window_manager: 预处理窗口管理器 若设置了该参数则会在调用 enterFrame（）
#       函数的时候将捕获的图像显示在指定的窗体上
#     should_mirror_preview：是否在指定窗口上镜像显示(水平翻转)

    def __init__(self, capture, previewWindowManager = None,
                 shouldMirrorPreview = False):
        # mirrorPreview 进行镜面反转
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        #定义私有属性
        # 属性加单下划线或者双下划线可以表示私有变量
        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFileName = None
        self._videoFileName = None
        self._videoEncoding = None
        self._videoWriter = None
        #摄像头调用   通道和帧的设置   图像文件 以及 视频文件
        self._startTime = None
        self._frameElapsed = int(0)
        self._fpsEstimate = None

    @property # 只读属性
    #内置的@property装饰器Python负责将一种方法转换为属性调用。
    def channel(self):
        return self.channel

    @channel.setter
    def channel(self, value):
        if self._channel !=value:
            self._channel =value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
            # retrieve 适用于多个摄像头进行同步
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFileName is not None

    @property
    def isWritingVideo(self):
        return self._videoFileName is not None

    def enterFrame(self):
        # 检测之前是不是有其他的帧还未退出
        assert not self._enteredFrame,\
            "previous enterFrame() had no matching exitFrame()"
        if self._capture is not None:
            #捕获帧
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """绘制到窗口  写入文件  释放帧"""
        #检测是否还有其他的帧是可以获取的
        # the getter 可以检索并取回帧
        if self.frame is None:
            self._enteredFrame = False
            return
        #更新帧速度估算值以及相关变量
        if self._frameElapsed == 0:
            self._startTime = time.time()
        else:
            timeElaspsed = time.time() - self._startTime
            self._fpsEstimate = self._frameElapsed / timeElaspsed
        self._frameElapsed += 1

         # 如果指定了窗口管理器，则在窗口中显示图像
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:  # 想要返回镜像
                mirroredFrame = numpy.fliplr(self._frame).copy()
                # 使得矩阵x沿着垂直轴左右进行翻转
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self.frame)

        #是否指定图片路径，将图片写入到文件中
        if self.isWritingImage:
            cv2.imwrite(self._imageFileName, self._frame)
            self._imageFileName = None

         #将每一帧图像写到视频文件中
        self._WriteVideoFrame()  #_WriteVideoFrame
         # 清空标志位 释放帧
        self._frame = None
        self._enteredFrame = False

    # 指定录入的图像路径
    def writeImage(self, filename):
        """指定每一帧图像的写入路径， 
        实际的写入操作会推迟到"下一次调用exitFrame 函数"""
        self._imageFilename = filename

    def startWritingVideo(self, filename,
                          encoding = cv2.VideoWriter_fourcc("I", "4", "2", "0")):
        self._videoFileName = filename
        self._videoEncoding = encoding

    #结束录制
    def stopWritingVideo(self):
        self._videoEncoding = None
        self._videoFilename = None
        self._videoWriter = None
# 录制视频  非公有函数

    def _WriteVideoFrame(self):
        if not self.isWritingVideo:
             return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                if self._framesElapsed < 20:
                    # 等待直到更多的帧经过 ，这样的话估算就比较容易和合适
                    return
                else:
                    fps = self._fps_Estimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFileName, self._videoEncoding,
                                                     fps, size)
        self._videoWriter.write(self._frame)