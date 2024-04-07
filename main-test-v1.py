# -*- coding:utf-8 -*-
# 用于手势识别，识别数字
import cv2
import serial
import time
from HandTrackingModule import HandDetector

class Main:
    def __init__(self):
        self.camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)# 摄像头捕获
        self.camera.set(3, 1280)#分辨率
        self.camera.set(4, 720)
        self.detector = HandDetector()  # 调用hand模块
    # 手势识别的函数
    def Gesture_recognition(self):
        start_time = time.time()

        ser = serial.Serial('COM10', 115200, timeout=1)
        if ser.isOpen():
            print(f"串口 {'COM5'} 打开成功")
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            frame, img = self.camera.read()#捕获摄像头输入
            img = self.detector.findHands(img)# 调用findhand函数
            lmList, bbox = self.detector.findPosition(img)# 手势识别 lmlist关节位置方位bbox为方框

            if lmList:# 如果非空，返回TRUE
                x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
                x1, x2, x3, x4, x5 = self.detector.fingersUp()
                if elapsed_time >= 0.05:
                    if (x2 == 0 and x3 == 0) and (x4 == 1 and x5 == 1 and x1 == 1):
                        cv2.putText(img, "2_TWO", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
                        ser.write(b'\x02')
                        print("2")
                    elif (x2 == 0 and x3 == 0 and x4 == 0) and (x1 == 1 and x5 == 1):
                        cv2.putText(img, "3_THREE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
                        ser.write(b'\x03')
                        print("3")
                    elif (x2 == 0 and x3 == 0 and x4 == 0 and x1 == 0) and (x5 == 1):
                        cv2.putText(img, "4_FOUR", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
                        # data = b'\x01'
                        # char_data = chr(data[0])
                        # ser.write(char_data.encode())
                        # data = b'\x01'
                        # hex_data = data.hex()
                        # hex_bytes = bytes.fromhex(hex_data)
                        # ser.write(hex_bytes)
                        # print("1")
                        # print("4")
                    elif x1 == 0 and x2 == 0 and x3 == 0 and x4 == 0 and x5 == 0:
                        cv2.putText(img, "5_FIVE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
                        # data = b'\x00'
                        # hex_data = data.hex()
                        # hex_bytes = bytes.fromhex(hex_data)
                        # ser.write(hex_bytes)
                        # data = b'\x00'
                        # char_data = chr(data[0])
                        # ser.write(char_data.encode())
                        # print(type(char_data))

                    elif x2 == 0 and (x1 == 1 and x3 == 1 and x4 == 1 and x5 == 1):
                        cv2.putText(img, "1_ONE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)

                    elif x1 == 0 and (x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1):
                        cv2.putText(img, "GOOD!", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
                        # ser.write(b'\x00')
                        print("0")
                    elif (x1 == 0 and x2 == 0 and x5 == 0) and (x3 == 1 and x4 == 1):
                        cv2.putText(img, "ROCK", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 255), 3)
                        # ser.write(b'\x03')
                        print("6")
                    start_time = current_time
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