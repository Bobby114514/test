# scripts/my_script.py

import cv2
import numpy as np

# 读取视频文件
video_path = 'github/opencv_project/video_pkg/stream.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频信息
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义视频写入对象
output_video_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 同时展示原始视频
    cv2.imshow('Original Video', frame)

    # 转换颜色空间为HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义蓝色的HSV范围
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # 根据颜色范围创建掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 通过掩码提取颜色范围内的像素
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 二值化
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 高斯滤波
    blurred = cv2.GaussianBlur(binary, (23, 23), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 处理轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 符合条件的轮廓，可能是装甲板
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 计算角度
        angle_rad = np.arctan2(h, w)
        angle_deg = np.degrees(angle_rad)

        # 显示角度信息(注释行是把角度信息写在每帧画面上)
        #cv2.putText(frame, f'Angle: {angle_deg:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,cv2.LINE_AA)
        print("radian:",angle_rad,"degree:",angle_deg)

    # 将修改后的帧写入输出视频
    out.write(frame)

    # 显示结果
    cv2.imshow('Armor Detection', frame)

    # 退出键盘事件
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
