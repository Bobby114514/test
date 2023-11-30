# scripts/my_script.py

import cv2
import numpy as np
import os

def process_image(image_path, output_folder):
    # 读取输入图像
    image = cv2.imread(image_path)

    # 转换颜色空间为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围（大致）
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # 根据颜色范围创建掩码
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 选择最大的轮廓
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        # 获取包围框
        x, y, w, h = cv2.boundingRect(max_contour)

        # 在原图上绘制矩形框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 进行仿射变换
        target_angle_degrees = 30  # 你需要设置旋转的角度
        center = (x + w // 2, y + h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, target_angle_degrees, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # 构造输出图像路径
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_rotated.jpg")

        # 保存处理结果
        cv2.imwrite(output_image_path, rotated_image)
        print(f"Processed image saved at: {output_image_path}")
    else:
        print(f"No valid contours found in {image_path}")

def main():
    # 图片文件夹路径
    folder_path = "test/opencv_project/video_pkg"
    
    # 输出文件夹路径
    output_folder = "test/opencv_project/video_pkg_output"
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历文件夹中的所有图片
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            process_image(image_path, output_folder)

if __name__ == "__main__":
    main()


