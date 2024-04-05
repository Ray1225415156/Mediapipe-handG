# -*- coding:utf-8 -*-
import cv2
import serial
import time

from HandTrackingModule import HandDetector


class Main:
    def __init__(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 摄像头捕获
        self.camera.set(3, 1280)  # 分辨率
        self.camera.set(4, 720)
        self.detector = HandDetector()  # 调用hand模块  `

    # 手势识别的函数
    def Gesture_recognition(self):
        start_time = time.time()
        ser = serial.Serial('COM4', 115200, timeout=1)
        if ser.isOpen():
            print(f"串口 {'COM5'} 打开成功")
        joint_side2 = [0, 0]
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            frame, img = self.camera.read()  # 捕获摄像头输入
            img = self.detector.findHands(img)  # 调用findhand函数
            lmList, bbox = self.detector.findPosition(img)  # 手势识别 lmlist关节位置方位bbox为方框

            if lmList:  # 如果非空，返回TRUE
                x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
                joint_state = self.detector.fingersUp()
                joint_side1 = self.detector.handSide()
                #张开为1，收紧为0
                if joint_side1[1] > 10 and joint_state[1] == 0:
                    joint_side2[0] = 1
                else:
                    joint_side2[0] = 0
                if joint_side1[3] > 15 and joint_state[4] == 0:
                    joint_side2[1] = 1
                else:
                    joint_side2[1] = 0
                combined_array = joint_state + joint_side2
                text_array = ''.join(str(x) for x in combined_array)
                # # 将整个列表转换为十六进制字符串
                # hex_data = ''.join(format(x, '02x') for x in combined_array)
                # # 将十六进制字符串转换为字节串并写入串口
                # hex_bytes = bytes.fromhex(hex_data)
                # # 在字节串的开头添加 0xFF
                # hex_bytes = b'\xFF' + hex_bytes
                # # 在字节串的末尾添加 0xFE
                # hex_bytes += b'\xFE'
                if elapsed_time >= 0.001:
                    print(text_array)
                    ser.write(text_array.encode('utf-8'))
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
