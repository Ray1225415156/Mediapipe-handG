import cv2
import numpy as np

import sys


# 打开摄像头
def reco():
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.5
    while cap.isOpened():
        # ---------图像处理-----------
        # 调节图像尺寸
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        # 高斯处理，提高环境适应性
        blur = cv2.blur(frame, (5, 5))
        blur0 = cv2.medianBlur(blur, 5)
        blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
        blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)

        # 转换为HSV图像
        hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)

        # 定义最高最低绿色HSV数值
        low_green = np.array([35, 43, 46])
        high_green = np.array([77, 255, 255])
        # 将定义在最低最高HSV值内图像保留，其余用黑色覆盖
        mask1 = cv2.inRange(hsv, low_green, high_green)

        # 利用canny算法对图像进行轮廓提取
        Canny = cv2.Canny(mask1, 20, 150)

        # 异常处理，防止在画面中无绿色物体出现时报错
        try:
            # 在提取出的轮廓图像中找出轮廓线条,并在原图上面画出矩阵框
            (cnts1, _) = cv2.findContours(Canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = sorted(cnts1, key=cv2.contourArea, reverse=True)[0]

            # 计算最大轮廓的旋转边界框
            rect = cv2.minAreaRect(c)

            # box里保存的是绿色矩形区域四个顶点的坐标(从边界框中提取出顶点坐标)
            box = np.int0(cv2.boxPoints(rect))

            # 将box的顶点坐标绘制在图中并连接线条
            cv2.drawContours(frame, [box], -1, (0, 0, 255), 5)
        except:
            pass
        frame=cv2.resize(frame,(800,600))
        cv2.putText(frame, "color:green", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2)
        cv2.imshow('reco', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
