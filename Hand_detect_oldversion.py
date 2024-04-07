import cv2
import numpy as np
import mediapipe as mp
import serial
#用于测试
# ser = serial.Serial('COM3', 9600, timeout=1)  # 不插入串口设备无法执行代码！！！！！！

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
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())  # 绘制关节点
        # 监测到右手，执行
        if results.right_hand_landmarks:
            RHL = results.right_hand_landmarks
            # 计算角度
            i = 0
            # ser.write("B0,".encode("utf8"))  # 用于体感手套控制机械手的老代码进行检测
            for joint in joint_list:  # 生成向量
                a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
                b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
                c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
                # 计算弧度
                radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度
                if angle > 180.0:  # 防止角度超出范围
                    angle = 360 - angle
                joint_angle[i] = round(angle, 1)
                cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)  # 将角度值放置在关节点旁边
                i = i + 1
            # ser.write("0E\r\n".encode("utf8"))  # 用于体感手套控制机械手的老代码进行检测
            print(joint_angle[0], joint_angle[1], joint_angle[2], joint_angle[3], joint_angle[4], joint_angle[5],
                  joint_angle[6], joint_angle[7], joint_angle[8], joint_angle[9], joint_angle[10])

        # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        cv2.imshow('Mediapipe Holistic', image)  # 取消镜面翻转
        if cv2.waitKey(5) == ord('q'):
            break
cap.release()
