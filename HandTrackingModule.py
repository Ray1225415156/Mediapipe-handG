# -*- coding:utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.modelComplex = modelComplexity
        self.mpHands = mp.solutions.hands  # mp的手部支持模块
        # Hands完成对Hands初始化配置
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.minTrackCon)
        # mode为True为图片输入
        # maxhand 为最大手数目
        # modelcomplex为模型的复杂度
        # detectioncon和trackcon为置信度阈值，越大越准确
        self.mpDraw = mp.solutions.drawing_utils  # 用于绘制
        self.tipIds = [4, 8, 12, 16, 20]  # 对应手指的指尖
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将传入的图转为RGB模式，
        # 返回一个列表，包含21个标志点的x、y、z的值
        self.results = self.hands.process(imgRGB)  # 完成图像的处理，输入必须为RGB格式的图片

        if self.results.multi_hand_landmarks:  # hands in list
            for handLms in self.results.multi_hand_landmarks:  # 提取每个指头并绘制标点和连接线
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
#注意返回值self.lmlist[][]存储的是关键点x，y，z的相对坐标
#维度1是关键点的编号，参考图片figure。维度2分别是x，y，z对应的坐标
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        zList = []  # 添加 z 轴坐标列表
        bbox = []
        bboxInfo = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                px, py = int(lm.x * w), int(lm.y * h)

                # 假设手势识别模型提供了 z 坐标在 landmark 结构中的表示方式
                pz = lm.z

                xList.append(px)
                yList.append(py)
                zList.append(pz)  # 将 z 坐标添加到列表中
                self.lmList.append([px, py, pz])  # 将 x、y、z 坐标添加到 lmList 中
                if draw:
                    cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
            bboxInfo = {"id": id, "bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (0, 255, 0), 2)

        return self.lmList, bboxInfo

    def fingersUp(self):
        if self.results.multi_hand_landmarks:
            myHandType = self.handType()
            fingers = []
            # Thumb
            if myHandType == "Right":
                if self.lmList[self.tipIds[0]][0] < self.lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if self.lmList[self.tipIds[0]][0] < self.lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][1] > self.lmList[self.tipIds[id] - 2][1]:
                    if self.lmList[self.tipIds[id] - 2][1] < self.lmList[self.tipIds[id] - 3][1]:
                        fingers.append(1)
                    elif self.lmList[self.tipIds[id] - 2][1] > self.lmList[self.tipIds[id] - 3][1]:
                        fingers.append(2)
                else:
                    fingers.append(0)
        return fingers

    # 判别左右手
    def handType(self):
        if self.results.multi_hand_landmarks:
            if self.lmList[17][0] < self.lmList[5][0]:
                return "Right"
            else:
                return "Left"

    # 角度判别和z坐标判别
    # 返回数据为角度值、z坐标值（均为按手指分类的二维数组）
    def handAngle(self):
        i = 0
        j = 0
        joint_list1 = [[19, 18, 17], [18, 17, 0], [15, 14, 13], [14, 13, 0], [11, 10, 9], [10, 9, 0], [7, 6, 5],
                       [6, 5, 0], [4, 3, 2], [3, 2, 1], [2, 1, 0]]  # 手指弯曲关节序列
        joint_list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # 手指关节序列
        # 创建一个空的二维数组存储角度值
        joint_angle_1 = [[0, 0] for _ in range(len(joint_list1))]  # 手指弯曲关节角度
        joint_z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if self.results.multi_hand_landmarks:
            myHandType = self.handType()
            if myHandType == "Right":
                for idx, joint_1 in enumerate(joint_list1):  # 弯曲角度
                    a = np.array([self.lmList[joint_1[0]][0], self.lmList[joint_1[0]][1]])
                    b = np.array([self.lmList[joint_1[1]][0], self.lmList[joint_1[1]][1]])
                    c = np.array([self.lmList[joint_1[2]][0], self.lmList[joint_1[2]][1]])
                    radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle_1 = np.abs(radians_fingers * 180.0 / np.pi)
                    if angle_1 > 180.0:
                        angle_1 = 360 - angle_1
                    # 将弧度值赋给二维数组
                    joint_angle_1[i] = round(angle_1, 1)  # 按照指定格 式赋值
                    i = i + 1
                for idx, joint_2 in enumerate(joint_list2):  # z坐标
                    z0 = self.lmList[0][2]
                    z = self.lmList[joint_2][2]
                    joint_z[j] = z - z0
                    j = j + 1
        # 将joint_angle、joint_z中的值调整为二维数组，按照手指类型进行归类
        categorized_joint_angle = []
        categorized_joint_z = []
        for finger_idx in range(5):
            finger_angles = [joint_angle_1[i] for i in range(finger_idx * 2, finger_idx * 2 + 2)]
            finger_z = [joint_z[i] for i in range(finger_idx * 4 + 1, finger_idx * 4 + 5)]
            categorized_joint_angle.append(finger_angles)
            categorized_joint_z.append(finger_z)
        return categorized_joint_angle, categorized_joint_z

    def handSide(self):
        n = 0
        joint_list = [[4, 2, 8], [8, 0, 9], [12, 9, 16], [13, 0, 20]]  # 手指测摆关节序列
        joint_angle_2 = [0, 0, 0, 0]  # 手指测摆关节角度
        if self.results.multi_hand_landmarks:
            myHandType = self.handType()
            if myHandType == "Right":
                for idx, joint in enumerate(joint_list):  # 侧摆角度
                    x = np.array([self.lmList[joint[0]][0], self.lmList[joint[0]][1]])
                    y = np.array([self.lmList[joint[1]][0], self.lmList[joint[1]][1]])
                    z = np.array([self.lmList[joint[2]][0], self.lmList[joint[2]][1]])
                    radians_fingers = np.arctan2(z[1] - y[1], z[0] - y[0]) - np.arctan2(x[1] - y[1], x[0] - y[0])
                    angle_2 = np.abs(radians_fingers * 180.0 / np.pi)
                    if angle_2 > 180.0:
                        angle_2 = 360 - angle_2
                    joint_angle_2[n] = round(angle_2, 1)  # 按照指定格 式赋值

                    n = n + 1
        return joint_angle_2
    # --旧版无分类--
    # def handAngle(self):
    #     i = 0
    #     j = 0
    #     joint_list1 = [[19, 18, 17], [18, 17, 0], [15, 14, 13], [14, 13, 0], [11, 10, 9], [10, 9, 0], [7, 6, 5],
    #                    [6, 5, 0], [4, 3, 2], [3, 2, 1], [2, 1, 0]]  # 手指关节序列
    #     joint_list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 手指关节序列
    #
    #     # 创建一个空的二维数组存储角度值
    #     joint_angle = [[0, 0, 0] for _ in range(len(joint_list1))]  # 手指关节角度
    #
    #     joint_z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #
    #     if self.results.multi_hand_landmarks:
    #         myHandType = self.handType()
    #         if myHandType == "Right":
    #             for idx, joint_1 in enumerate(joint_list1):  # 使用enumerate获取索引以及值
    #                 a = np.array([self.lmList[joint_1[0]][0], self.lmList[joint_1[0]][1]])
    #                 b = np.array([self.lmList[joint_1[1]][0], self.lmList[joint_1[1]][1]])
    #                 c = np.array([self.lmList[joint_1[2]][0], self.lmList[joint_1[2]][1]])
    #                 radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    #                 angle = np.abs(radians_fingers * 180.0 / np.pi)
    #                 if angle > 180.0:
    #                     angle = 360 - angle
    #                 # 将弧度值赋给二维数组
    #                 joint_angle[i] = round(angle, 1), 0 # 按照指定格式赋值
    #                 i = i + 1
    #             for idx, joint_2 in enumerate(joint_list2):
    #                 z0 = self.lmList[0][2]
    #                 z = self.lmList[joint_2][2]
    #                 joint_z[j] = z - z0
    #                 j = j + 1
    #     return joint_angle, joint_z
