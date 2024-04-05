skel_line_conf = [(11, 12), (11, 13),
                  (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                  (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                  (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                  (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                  (29, 31), (30, 32), (27, 31), (28, 32)]


def draw_skeleton(skeleton_data, frame):
    W = frame.shape[0]
    H = frame.shape[1]
    # 33个点的坐标
    skeleton_data = skeleton_data.landmark
    compare_list = []
    # to draw the lines joining the points
    for idx in range(len(skel_line_conf)):
        # 取出两个点的坐标
        first_xy = skel_line_conf[idx][0]
        second_xy = skel_line_conf[idx][1]

        x1 = int(skeleton_data[first_xy].x * H)
        y1 = int(skeleton_data[first_xy].y * W)

        x2 = int(skeleton_data[second_xy].x * H)
        y2 = int(skeleton_data[second_xy].y * W)
        # 计算斜率以及允许偏差
        compare_list.append((y1 - y2) / ((x1 - x2) + 1e-6))
    # 返回斜率值
    return compare_list
