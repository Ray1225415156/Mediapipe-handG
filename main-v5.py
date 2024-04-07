# 依据角度分级，需侧向放手，共5级
# 未测试
import cv2
import serial
import time
import numpy as np
from HandTrackingModule import HandDetector


class Main:
    def __init__(self):
        self.detector = HandDetector()
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 摄像头捕获
        self.camera.set(3, 1280)  # 分辨率
        self.camera.set(4, 720)

    # 手势识别的函数
    def Gesture_recognition(self):
        start_time = time.time()
        joint_state = [0, 0, 0, 0, 0]
        joint_side2 = [0, 0]
        # ser = serial.Serial('COM6', 9600, timeout=1)
        # if ser.isOpen():
        #     print(f"串口 {'COM5'} 打开成功")
        while True:
            current_time = time.time()  # 用于时间轴
            elapsed_time = current_time - start_time
            frame, img = self.camera.read()  # 捕获摄像头输入
            img = self.detector.findHands(img)  # 调用findhand函数
            lmList, bbox = self.detector.findPosition(img)  # 手势识别 lmlist关节位置方位bbox为方框
            if lmList:  # 如果非空，返回TRUE
                angle, joint_angle_2 = self.detector.handAngle()
                joint_side1 = self.detector.handSide()
                # print(joint_angle_2)
                if elapsed_time >= 1:
                    for i in range(5):
                        if int(angle[i][0]) >= 170 and int(angle[i][1]) >= 170:
                            joint_state[i] = 0  # 伸直状态
                        elif (150 <= int(angle[i][0]) <= 170) and (int(angle[i][1]) >= 170):
                            joint_state[i] = 1  # 半弯曲状态
                        elif (120 <= int(angle[i][0]) <= 150) and (int(angle[i][1]) >= 170):
                            joint_state[i] = 2  # 半弯曲状态
                        elif (90 <= int(angle[i][0]) <= 120) and (int(angle[i][1]) >= 170):
                            joint_state[i] = 3  # 半弯曲状态
                        elif (60 <= int(angle[i][0]) <= 90) and (130 <= int(angle[i][1]) <= 170):
                            joint_state[i] = 4  # 半弯曲状态
                        elif int(angle[i][1]) <= 90:
                            joint_state[i] = 5  # 全弯曲状态
                    if joint_side1[1] > 10 and joint_state[1] == 0:
                        joint_side2[0] = 1
                    else:
                        joint_side2[0] = 0
                    if joint_side1[3] > 15 and joint_state[4] == 0:
                        joint_side2[1] = 1
                    else:
                        joint_side2[1] = 0
                    combined_array = joint_state + joint_side2
                    start_time = current_time
                    print(combined_array)
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
