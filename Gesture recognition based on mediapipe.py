import cv2
import numpy as np
import mediapipe as mp
import serial
import time
# import threading
# 发送数据函数
# def send_data():
#     while True:
#         ser.write(b'1')  # 发送数据给单片机
#         print('数据已发送')
#         threading.Timer(1, send_data).start()  # 每隔1秒触发一次发送数据函数
start_time = time.time()
port = 'COM6'
baud_rate = 9600
timeout = 1
result1 = 1
try:
    # ser = serial.Serial(port, baud_rate, timeout=timeout)
    # if ser.isOpen():
    #     print(f"串口 {port} 打开成功")
    # print(start_time)
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
            if results.left_hand_landmarks:
                RHL = results.left_hand_landmarks
                # 计算角度
                i = 0
                for joint in joint_list:  # 生成向量ef
                    a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
                    b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
                    c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
                    # 计算弧度
                    radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度
                    if angle > 180.0:  # 防止角度超出范围
                        angle = 360 - angle
                    joint_angle[i] = round(angle, 1)
                    # print(elapsed_time)
                    if i == 1:  # 每个人手不一样，可以自己调节
                        if joint_angle[i] > 0:
                            result2 = f"{result1:02X}"
                            result3 = bytes.fromhex(result2)
                    cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)  # 将角度值放置在关节点旁边
                    i = i + 1
                # print(joint_angle[0], joint_angle[1], joint_angle[1], joint_angle[1], joint_angle[1], joint_angle[1], joint_angle[1], joint_angle[1], )

            # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            cv2.imshow('Mediapipe Holistic', image)  # 取消镜面翻转
            if cv2.waitKey(5) == ord('q'):
                data = ser.read(1)
                print(data)
                break
    cap.release()
except serial.SerialException as e:
    print(f"串口 {port} 打开失败：{str(e)}")
except Exception as e:
    print(f"发生异常：{str(e)}")
