# -*- coding:utf-8 -*-
import cv2
from HandTrackingModule import HandDetector

class Main:
    def __init__(self):
        self.camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)# 摄像头捕获
        self.camera.set(3, 1280)#分辨率
        self.camera.set(4, 720)
    # 手势识别的函数
    def Gesture_recognition(self):
        while True:
            self.detector = HandDetector()# 调用hand模块
            frame, img = self.camera.read()#捕获摄像头输入
            img = self.detector.findHands(img)# 调用findhand函数
            lmList, bbox = self.detector.findPosition(img)# 手势识别 lmlist关节位置方位bbox为方框

            if lmList:# 如果非空，返回TRUE
                x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
                x1, x2, x3, x4, x5 = self.detector.fingersUp()

                if (x2 == 1 and x3 == 1) and (x4 == 0 and x5 == 0 and x1 == 0):
                    cv2.putText(img, "2_TWO", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)

                elif (x2 == 1 and x3 == 1 and x4 == 1) and (x1 == 0 and x5 == 0):
                    cv2.putText(img, "3_THREE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)

                elif (x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1) and (x1 == 0):
                    cv2.putText(img, "4_FOUR", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)

                elif x1 == 1 and x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1:
                    cv2.putText(img, "5_FIVE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)

                elif x2 == 1 and (x1 == 0, x3 == 0, x4 == 0, x5 == 0):
                    cv2.putText(img, "1_ONE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)

                elif x1 and (x2 == 0, x3 == 0, x4 == 0, x5 == 0):
                    cv2.putText(img, "GOOD!", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)

            cv2.imshow("camera", img)# 显示图片
            # 点击窗口关闭按钮退出程序
            if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
                break
            # 点击小写字母q 退出程序
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# main
if __name__ == '__main__':
    Solution = Main()
    Solution.Gesture_recognition()