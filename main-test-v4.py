# -*- coding:utf-8 -*-
import cv2
import serial
import time
import random
from HandTrackingModule import HandDetector


class Main:
    def __init__(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 摄像头捕获
        self.camera.set(3, 1280)  # 分辨率
        self.camera.set(4, 720)
        self.detector = HandDetector()  # 调用hand模块

    # 手势识别的函数
    def Gesture_recognition(self):
        start_time = time.time()
        ser = serial.Serial('COM4', 115200, timeout=1)
        if ser.isOpen():
            print(f"串口 {'COM5'} 打开成功")
        scissor_1 = [0, 0, 0, 2, 2]
        cloth_1 = [0, 0, 0, 0, 0]
        stone_1 = [0, 2, 2, 2, 2]
        scissor_2 = [1, 0, 0, 2, 2]
        cloth_2 = [0, 0, 0, 0, 0]
        stone_2 = [1, 2, 2, 2, 2]
        scissor = ''.join(str(x) for x in scissor_1)
        cloth = ''.join(str(x) for x in cloth_1)
        stone = ''.join(str(x) for x in stone_1)
        scissor2 = ''.join(str(x) for x in scissor_2)
        cloth2 = ''.join(str(x) for x in cloth_2)
        stone2 = ''.join(str(x) for x in stone_2)
        state = 0;#石头0 剪刀1 布2
        flag = 1;#随机0 必输1 必赢2
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            frame, img = self.camera.read()  # 捕获摄像头输入
            img = self.detector.findHands(img)  # 调用findhand函数
            lmList, bbox = self.detector.findPosition(img)  # 手势识别 lmlist关节位置方位bbox为方框
            if lmList:  # 如果非空，返回TRUE
                x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
                try:
                    x1, x2, x3, x4, x5 = self.detector.fingersUp()
                except ValueError:
                    # 处理值不足的情况
                    x1, x2, x3, x4 = self.detector.fingersUp()
                x_array = x1, x2, x3, x4
                if flag == 0:
                    # if elapsed_time % 3 <= 0.001:
                    #     x1_old = x1,x2,x3,x4
                    #     if x_array != x1_old:
                    #         state = random.randint(0, 2)  # 生成 1 到 10 之间的随机整数
                elif flag == 1:
                    if (x2 and x3) == 0 and (x1 and x4 and x5) != 0:
                        state = 0
                    elif (x2 and x3 and x1 and x4 and x5) == 0:
                        state = 1
                    elif (x2 and x3 and x1 and x4 and x5) == 1 or 2:
                        state = 2
                elif flag == 2:
                    if (x2 and x3) == 0 and (x1 and x4 and x5) != 0:
                        state = 2
                    elif (x2 and x3 and x1 and x4 and x5) == 0:
                        state = 0
                    elif (x2 and x3 and x1 and x4 and x5) == 1 or 2:
                        state = 1
                print(state)
                if int(elapsed_time*100) % 5 <= 1:
                    if state == 0:
                        ser.write(stone.encode('utf-8'))
                    elif state == 1:
                        ser.write(scissor.encode('utf-8'))
                    elif state == 2:
                        ser.write(cloth.encode('utf-8'))
                if elapsed_time == 0.1:
                    if state == 0:
                        ser.write(stone2.encode('utf-8'))
                    elif state == 1:
                        ser.write(scissor2.encode('utf-8'))
                    elif state == 2:
                        ser.write(cloth2.encode('utf-8'))
                    start_time = current_time

            cv2.imshow("camera", img)  # 显示图片
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
