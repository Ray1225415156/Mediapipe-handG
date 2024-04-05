import cv2
import numpy as np
import mediapipe as mp
import time

start_time = time.time()
timeout = 1
result1 = 1
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

joint_list = [[19, 18, 17], [18, 17, 0], [15, 14, 13], [14, 13, 0], [11, 10, 9], [10, 9, 0], [7, 6, 5], [6, 5, 0],
              [4, 3, 2], [3, 2, 1], [2, 1, 0]]  # 手指关节序列
joint_angle = [180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180]  # 手指关节角度
cap = cv2.VideoCapture(0)  # 开启电脑摄像头

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()  # 截取摄像头中的一帧画面
        if not success:
            print("Ignoring empty camera frame.")
            break
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 对图像进行色彩空间的转换
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转换回来
        # 渲染
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.face_landmarks,
        #     mp_holistic.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_tesselation_style())
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())  # 绘制关节点
        # 监测到右手，执行
        if results.right_hand_landmarks:
            RHL = results.right_hand_landmarks
            current_time = time.time()
            # print(current_time)
            elapsed_time = current_time - start_time
            # 计算角度
            i = 0
            # ser.write("B0,".encode("utf8"))  # 用于体感手套控制机械手的老代码进行检测
            for joint in joint_list:  # 生成向量
                a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
                b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
                c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
                cz1 = np.array([RHL.landmark[joint[0]].z, RHL.landmark[joint[1]].z, RHL.landmark[joint[2]].z])
                # 计算弧度
                radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度
                if angle > 180.0:  # 防止角度超出范围
                    angle = 360 - angle
                joint_angle[i] = round(angle, 1)
                # print(elapsed_time)
                # if i == 1:  # 每个人手不一样，可以自己调节
                #     if joint_angle[i] > 0:
                #         result2 = f"{result1:02X}"
                #         result3 = bytes.fromhex(result2)
                #         if elapsed_time >= 1:
                #             ser.write(b'\x02')
                #             # ser.read(10)
                #             print("1")
                #             start_time = current_time
                # elif 50 < joint_angle[i] < 90:
                #     ser.write(1)
                # elif 95 < joint_angle[i] < 135:
                #     ser.write(2)
                # elif 140 < joint_angle[i] < 160:
                #     ser.write(3)
                # elif 165 < joint_angle[i] < 180:
                #     ser.write(4)
                # elif 110 < joint_angle[i] < 120:
                #     ser.write(5)
                # elif 125 < joint_angle[i] < 135:
                #     ser.write(6)
                # elif 140 < joint_angle[i] < 150:
                #     ser.write(7)
                # elif 155 < joint_angle[i] < 165:
                #     ser.write(8)
                # elif 170 < joint_angle[i] < 180:
                #     ser.write(9)
                # elif i == 3:
                # if joint_angle[i] < 50:
                #     ser.write(0)
                # elif 55 < joint_angle[i] < 65:
                #     ser.write(1)
                # elif 69 < joint_angle[i] < 79:
                #     ser.write(2)
                # elif 84 < joint_angle[i] < 94:
                #     ser.write(3)
                # elif 98 < joint_angle[i] < 108:
                #     ser.write(4)
                # elif 113 < joint_angle[i] < 123:
                #     ser.write(5)
                # elif 127 < joint_angle[i] < 137:
                #     ser.write(6)
                # elif 142 < joint_angle[i] < 152:
                #     ser.write(7)
                # elif 156 < joint_angle[i] < 166:
                #     ser.write(8)
                # elif 171 < joint_angle[i] < 180:
                #     ser.write(9)
                # elif i == 5:
                #     if joint_angle[i] < 60:
                #         ser.write(0)
                #     elif 64 < joint_angle[i] < 73:
                #         ser.write(1)
                #     elif 77 < joint_angle[i] < 86:
                #         ser.write(2)
                #     elif 90 < joint_angle[i] < 99:
                #         ser.write(3)
                #     elif 103 < joint_angle[i] < 112:
                #         ser.write(4)
                #     elif 116 < joint_angle[i] < 125:
                #         ser.write(5)
                #     elif 129 < joint_angle[i] < 138:
                #         ser.write(6)
                #     elif 142 < joint_angle[i] < 151:
                #         ser.write(7)
                #     elif 155 < joint_angle[i] < 165:
                #         ser.write(8)
                #     elif 169 < joint_angle[i] < 178:
                #         ser.write(9)
                # elif i == 7:
                #     if joint_angle[i] < 80:
                #         ser.write(0)
                #     elif 83 < joint_angle[i] < 91:
                #         ser.write(1)
                #     elif 94 < joint_angle[i] < 102:
                #         ser.write(2)
                #     elif 105 < joint_angle[i] < 113:
                #         ser.write(3)
                #     elif 116 < joint_angle[i] < 124:
                #         ser.write(4)
                #     elif 127 < joint_angle[i] < 135:
                #         ser.write(5)
                #     elif 138 < joint_angle[i] < 146:
                #         ser.write(6)
                #     elif 149 < joint_angle[i] < 157:
                #         ser.write(7)
                #     elif 160 < joint_angle[i] < 168:
                #         ser.write(8)
                #     elif 170 < joint_angle[i] < 180:
                #         ser.write(9)
                # elif i == 8:
                #     if joint_angle[i] < 70:
                #         ser.write(0)
                #     elif 74 < joint_angle[i] < 82:
                #         ser.write(1)
                #     elif 86 < joint_angle[i] < 94:
                #         ser.write(2)
                #     elif 98 < joint_angle[i] < 106:
                #         ser.write(3)
                #     elif 110 < joint_angle[i] < 118:
                #         ser.write(4)
                #     elif 122 < joint_angle[i] < 130:
                #         ser.write(5)
                #     elif 134 < joint_angle[i] < 142:
                #         ser.write(6)
                #     elif 146 < joint_angle[i] < 152:
                #         ser.write(7)
                #     elif 156 < joint_angle[i] < 164:
                #         ser.write(8)
                #     elif 168 < joint_angle[i] < 180:
                #         ser.write(9)
                # elif i == 10:
                #     if joint_angle[i] < 145:
                #         ser.write(0)
                #     elif 145 <= joint_angle[i] < 146:
                #         ser.write(1)
                #     elif 146 <= joint_angle[i] < 147:
                #         ser.write(2)
                #     elif 147 <= joint_angle[i] < 148:
                #         ser.write(3)
                #     elif 148 <= joint_angle[i] < 149:
                #         ser.write(4)
                #     elif 149 <= joint_angle[i] < 150:
                #         ser.write(5)
                #     elif 150 <= joint_angle[i] < 151:
                #         ser.write(6)
                #     elif 151 <= joint_angle[i] < 152:
                #         ser.write(7)
                #     elif 152 <= joint_angle[i] < 153:
                #         ser.write(8)
                #     elif 153 <= joint_angle[i]:
                #         ser.write(9)
                # ser.write(",".encode("utf8"))
                cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)  # 将角度值放置在关节点旁边
                i = i + 1
            # print(joint_angle[0], joint_angle[1], joint_angle[2], joint_angle[3], joint_angle[4], joint_angle[5], joint_angle[6], joint_angle[7],joint_angle[8],joint_angle[9],joint_angle[10])

        # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        cv2.imshow('Mediapipe Holistic', image)  # 取消镜面翻转
        if cv2.waitKey(5) == ord('q'):
            # data = ser.read(1)
            # print(data)
            break
cap.release()
# except serial.SerialException as e:
#     print(f"串口 {port} 打开失败：{str(e)}")
# except Exception as e:
#     print(f"发生异常：{str(e)}")
