
 # TechVidvan Human pose estimator
# import necessary packages

import cv2
import mediapipe as mp
import numpy as np
from utils_new import draw_skeleton
import matplotlib.pyplot as plt
import time
from reco import reco


def compare():
    # 初始化------------------------------------------------------------------------------------------
    # 实例化类方法，自上而下为：脸部识别、绘图工具、姿态估计、绘图样式
    mpFace = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles

    # 自定义人脸识别方法，最小的人脸检测置信度0.5
    faceDetection = mpFace.FaceDetection(min_detection_confidence=0.5)
    # 姿态识别方法，最小置信度为0.5
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    pose2 = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # 初始化变量，视频、分数、列表等
    cap = cv2.VideoCapture('cxk.mp4')
    cap2 = cv2.VideoCapture(0)
    count = 0
    sum = 0
    average = 0
    total = 0
    new_score = 50
    score_list = []
    boxlist = []  # 保存每帧图像每个框的信息
    pTime = 0  # 记录每帧图像处理的起始时间
    # 主循环开启------------------------------------------------------------------
    while cap.isOpened():
        # 读取图像，frame为图像信息，success表示是否成功
        success, frame = cap.read()
        # 如果启动不成功，打印end
        if not success:
            print("End.")
            break
        # 启动成功则计次加1
        count += 1
        success, frame2 = cap2.read()
        if not success:
            print("End.")
            break
        # 重新设置图像大小、翻转图像（竖直方向）
        frame = cv2.resize(frame, (600, 400))
        frame2 = cv2.resize(frame2, (600, 400))
        frame2 = cv2.flip(frame2, 1)
        # BGR转RGB
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        RGB2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        # 姿态对比---------------------------------------------------------------------------------
        # 获取斜率列表
        results = pose.process(RGB)
        com_list = draw_skeleton(results.pose_landmarks, frame)
        results2 = pose2.process(RGB2)
        com_list2 = draw_skeleton(results2.pose_landmarks, frame2)
        # 比较前12个向量，并依次计算出符合度分数
        if len(com_list) == len(com_list2):
            vec1 = np.array(com_list)[:12]
            vec2 = np.array(com_list2)[:12]
            score = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            score = (score + 1) / 2
            print(score)
            score_list.append(score)
        # 绘制骨架
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # 显示分数变化折线图
        sum = sum + score
        average = sum / count
        plt.plot(score_list, color='green', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Scores')
        plt.title('Histogram of Scores')
        # 人脸识别------------------------------------------------------------------------------------------
        # 将每一帧图像传给人脸识别模块
        result = faceDetection.process(RGB2)
        if result.detections:
            # 返回人脸索引index(第几张脸)，和关键点的坐标信息
            for index, detection in enumerate(result.detections):
                # 每帧图像返回一次是人脸的几率，以及识别框的xywh，后续返回关键点的xy坐标
                print(detection.score)  # 是人脸的的可能性
                print(detection.location_data.relative_bounding_box)  # 识别框的xywh

                # 设置一个边界框，接收所有的框的xywh及关键点信息
                bboxC = detection.location_data.relative_bounding_box

                # 接收每一帧图像的宽、高、通道数
                ih, iw, ic = frame2.shape

                # 将边界框的坐标点从比例坐标转换成像素坐标
                # 将边界框的宽和高从比例长度转换为像素长度
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                        int(bboxC.width * iw), int(bboxC.height * ih))

                # 把人脸的概率显示在检测框上,img画板，概率值*100保留两位小数变成百分数，再变成字符串
                cv2.putText(frame2, f'{str(round(detection.score[0] * 100, 2))}%',
                            (bbox[0], bbox[1] - 20),  # 文本显示的位置，-20是为了不和框重合
                            cv2.FONT_HERSHEY_PLAIN,  # 文本字体类型
                            2, (0, 0, 255), 2)  # 字体大小; 字体颜色; 线条粗细

                # 保存索引，人脸概率，识别框的x/y/w/h
                boxlist.append([index, detection.score, bbox])

                # （3）修改识别框样式
                x, y, w, h = bbox  # 获取识别框的信息,xy为左上角坐标点
                x1, y1 = x + w, y + h  # 右下角坐标点

                # 绘制比矩形框粗的线段，img画板，线段起始点坐标，线段颜色，线宽为8
                cv2.line(frame2, (x, y), (x + 20, y), (255, 0, 255), 4)
                cv2.line(frame2, (x, y), (x, y + 20), (255, 0, 255), 4)

                cv2.line(frame2, (x1, y1), (x1 - 20, y1), (255, 0, 255), 4)
                cv2.line(frame2, (x1, y1), (x1, y1 - 20), (255, 0, 255), 4)

                cv2.line(frame2, (x1, y), (x1 - 20, y), (255, 0, 255), 4)
                cv2.line(frame2, (x1, y), (x1, y + 20), (255, 0, 255), 4)

                cv2.line(frame2, (x, y1), (x + 20, y1), (255, 0, 255), 4)
                cv2.line(frame2, (x, y1), (x, y1 - 20), (255, 0, 255), 4)

                # 在每一帧图像上绘制矩形框
                cv2.rectangle(frame2, bbox, (255, 0, 255), 1)  # 自定义绘制函数

        # --------------------------------------------------------------------------------------

        # 记录每帧图像处理所花的时间
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # 计算fps值
        pTime = cTime  # 更新每张图像处理的初始时间

        # 拼接图像
        image1 = cv2.resize(frame, (800, 600))
        image2 = cv2.resize(frame2, (800, 600))
        result2show = np.hstack([image1, image2])
        cv2.imshow('Output', result2show)
        # 分数显示界面------------------------------------------------------
        image3 = cv2.imread("12.webp")
        image3 = cv2.resize(image3, (600, 400))
        cv2.putText(image3, "score:" + str(f"{score:.2f}"), (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255),
                    2)
        cv2.putText(image3, f'FPS: {str(int(fps))}', (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2)
        cv2.putText(image3, f'Probability:{str(round(detection.score[0] * 100, 2))}%', (150, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2)
        cv2.imshow('Score', image3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    cap2.release()
    while 1:
        # 结算界面---------------------------------------------------------------
        average = 1
        image4 = cv2.imread("123.jpg")
        image4 = cv2.resize(image4, (600, 400))
        total = average * 100
        cv2.putText(image4, "Finalscore:" + str(f"{total:.0f}"), (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                    (255, 255, 255), 2)
        cv2.putText(image4, "Finalscore:" + str(f"{total:.0f}"), (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                    (255, 255, 255), 2)
        if average >= 0.3 and average < 0.6:
            cv2.putText(image4, "Evaluate:good", (150, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 255), 2)
        elif average >= 0.6:
            cv2.putText(image4, "Evaluate:excellent", (150, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 2)
        else:
            cv2.putText(image4, "Evaluate:bad", (150, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
        cv2.imshow('Finalscore', image4)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    plt.show()
    cv2.destroyAllWindows()



image5 = cv2.imread("52.webp")
image5 = cv2.resize(image5, (600, 700))
cv2.putText(image5, "Select level", (190, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2)
cv2.putText(image5, "Attitude estimation and comparison(press a)", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
cv2.putText(image5, "Gesture recognition(press b)", (130, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
cv2.putText(image5, "Color Recognition(press c)", (140, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

while 1:
    cv2.imshow('Start', image5)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        cv2.destroyAllWindows()
        compare()
    if cv2.waitKey(1) & 0xFF == ord('b'):
        cv2.destroyAllWindows()
        import hand
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.destroyAllWindows()
        reco()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


