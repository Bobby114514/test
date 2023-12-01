##基于opencv定位装甲板（识别两侧红色灯带）
##基于卡尔曼滤波对装甲板移动进行预测
##已经安装了usb_cam包用于连接usb摄像头（假如是的话）

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ArmorTrackerROS:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('armor_tracker_node', anonymous=True)
        # 创建CvBridge
        self.bridge = CvBridge()
        # 初始化卡尔曼滤波器
        self.kalman_filter_init()

        # 设置红色阈值
        self.lower_red = np.array([0, 100, 100])
        self.upper_red = np.array([10, 255, 255])

        # 订阅图像消息
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        # 发布处理后的图像
        self.image_pub = rospy.Publisher('/armor_tracker/output_image', Image, queue_size=10)

    def kalman_filter_init(self):
        # 初始化卡尔曼滤波器参数
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 1e-5 * np.eye(4)
        self.kalman.measurementNoiseCov = 1e-1 * np.eye(2)

    def image_callback(self, msg):
        try:
            # 将ROS Image消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # 使用颜色阈值获取红色灯带的二进制掩码
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 假设选择最大的轮廓
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            # 计算装甲板中心
            center = np.array([x + w / 2, y + h / 2], dtype=np.float32)

            # 卡尔曼滤波预测
            prediction = self.kalman.predict()
            # 更新卡尔曼滤波器
            measurement = np.array([center], dtype=np.float32)
            self.kalman.correct(measurement)

            # 在图像上绘制装甲板
            cv2.rectangle(cv_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            # 在图像上绘制卡尔曼滤波的预测结果
            cv2.circle(cv_image, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)

        try:
            # 将处理后的图像转换为ROS Image消息并发布
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    try:
        tracker = ArmorTrackerROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
