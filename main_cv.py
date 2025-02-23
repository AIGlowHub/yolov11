import cv2
import torch
import numpy as np
from time import time
from ultralytics import YOLO

# 加载本地YOLOv11-pose模型
model = YOLO('yolo11m-pose.pt')  # 替换为你的模型文件路径
model.conf = 0.5  # 设置置信度阈值
model.iou = 0.5  # 设置IOU阈值

# 打开本地摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头可以尝试1,2等

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 设置高度
cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率

# 获取实际的摄像头参数（可能与设置值不同）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"摄像头信息：")
print(f"分辨率: {frame_width}x{frame_height}")
print(f"帧率: {fps}")

# 加载OpenPose姿态估计模型（可以选择适当的模型）
# 如果你使用OpenPose，通常需要安装openpose库，或者可以使用HRNet/AlphaPose等
# 在这个示例中，我们暂时假设有一个姿态估计的模型接口

# 定义YOLO-Pose的关键点连接（适用于17个关键点）
POSE_PAIRS = [
    [0, 1], [1, 3], [0, 2], [2, 4],  # 头部和耳朵
    [5, 6],  # 肩膀连接
    [5, 7], [7, 9], [6, 8], [8, 10],  # 手臂
    [5, 11], [6, 12],  # 躯干
    [11, 13], [13, 15], [12, 14], [14, 16]  # 腿部
]

# 定义关键点名称（17个关键点）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]


class PoseTracker:
    def __init__(self, history_size=10, time_window=2.0):  # time_window单位为秒
        self.history_size = history_size
        self.time_window = time_window
        self.position_history = []  # 存储历史位置
        self.height_history = []  # 存储身体高度历史
        self.time_history = []  # 存储时间戳
        self.last_state = "Standing"
        self.fall_start_time = None
        self.angle_history = []
        self.stable_count = 0

    def update_histories(self, center_up, body_height, angle):
        self.position_history.append(center_up)
        self.height_history.append(body_height)
        self.time_history.append(time())
        self.angle_history.append(angle)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
            self.height_history.pop(0)
            self.time_history.pop(0)
            self.angle_history.pop(0)


def detect_pose_state(keypoints, tracker):
    """
    使用几何方法检测摔倒：
    1. 计算躯干角度（肩部中心到髋部中心的线与垂直线的夹角）
    2. 检查人体框的宽高比
    3. 检查上下身中心点的相对位置
    """
    try:
        # 获取关键点坐标
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]

        # 计算上下身中心点
        center_up = (left_shoulder + right_shoulder) / 2
        center_down = (left_hip + right_hip) / 2

        # 计算直角三角形的第三个点（用于角度计算）
        right_angle_point = np.array([center_down[0], center_up[1]])

        # 计算三角形的两边
        a = abs(right_angle_point[0] - center_up[0])  # 水平距离
        b = abs(center_down[1] - right_angle_point[1])  # 垂直距离

        # 计算角度
        if b != 0:
            angle = np.degrees(np.arctan(a / b))
        else:
            angle = 90

        # 计算人体框的宽高比
        body_width = np.linalg.norm(left_shoulder - right_shoulder)
        body_height = np.linalg.norm(center_up - center_down)
        width_height_ratio = body_width / (body_height + 1e-6)

        # 更新跟踪器数据
        tracker.update_histories(center_up, body_height, angle)

        # 摔倒判断条件
        is_fallen = (
                angle > 60 or  # 角度大于60度
                center_down[1] <= center_up[1] or  # 上下身中心点位置异常
                width_height_ratio > 5 / 3  # 宽高比异常
        )

        if is_fallen:
            if tracker.state != "Fallen":
                tracker.fall_start_time = time()
            tracker.stable_count = 0
            tracker.state = "Fallen"
        else:
            tracker.stable_count += 1
            if tracker.stable_count > 10:  # 需要连续10帧才能恢复站立状态
                tracker.state = "Standing"
                tracker.fall_start_time = None

        return tracker.state

    except Exception as e:
        print(f"检测异常: {str(e)}")
        return "Error"


def visualize_skeleton(frame, keypoints, state, tracker):
    try:
        # 提高置信度阈值
        confidence_threshold = 0.5  # 提高置信度阈值

        # 绘制人体框
        if len(keypoints) >= 17:
            points = keypoints[:, :2][keypoints[:, 2] > confidence_threshold]
            if len(points) > 0:
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                cv2.rectangle(frame,
                              (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)),
                              (0, 0, 255), 2)

        # 更新骨架连接，添加手臂完整连接
        main_connections = [
            [5, 6],  # 肩膀连接
            [5, 11],  # 左躯干
            [6, 12],  # 右躯干
            [11, 12],  # 髋部连接
            [5, 7],  # 左上臂
            [7, 9],  # 左前臂
            [6, 8],  # 右上臂
            [8, 10],  # 右前臂
            [11, 13],  # 左大腿
            [12, 14],  # 右大腿
            [13, 15],  # 左小腿
            [14, 16]  # 右小腿
        ]

        # 绘制骨架
        for pair in main_connections:
            i, j = pair
            if (keypoints[i][2] > confidence_threshold and
                    keypoints[j][2] > confidence_threshold and
                    0 <= int(keypoints[i][0]) < frame.shape[1] and
                    0 <= int(keypoints[i][1]) < frame.shape[0] and
                    0 <= int(keypoints[j][0]) < frame.shape[1] and
                    0 <= int(keypoints[j][1]) < frame.shape[0]):

                pt1 = tuple(map(int, keypoints[i][:2]))
                pt2 = tuple(map(int, keypoints[j][:2]))

                # 根据不同部位使用不同颜色
                if i in [5, 6] and j in [7, 8]:  # 上臂
                    color = (0, 165, 255)  # 橙色
                elif i in [7, 8] and j in [9, 10]:  # 前臂
                    color = (0, 127, 255)  # 浅橙色
                else:
                    color = (0, 165, 255)  # 默认颜色

                cv2.line(frame, pt1, pt2, color, 2)

        # 更新关键点列表，包含所有重要关键点
        main_points = [
            5, 6,  # 肩膀
            7, 8,  # 肘部
            9, 10,  # 手腕
            11, 12,  # 髋部
            13, 14,  # 膝盖
            15, 16  # 脚踝
        ]

        # 绘制关键点
        for i in main_points:
            x, y, conf = keypoints[i]
            if (conf > confidence_threshold and
                    0 <= int(x) < frame.shape[1] and
                    0 <= int(y) < frame.shape[0]):

                # 根据不同部位使用不同颜色和大小
                if i in [9, 10]:  # 手腕
                    color = (255, 0, 0)  # 蓝色
                    size = 6
                elif i in [7, 8]:  # 肘部
                    color = (255, 127, 0)  # 浅蓝色
                    size = 5
                else:  # 其他关键点
                    color = (255, 0, 0)
                    size = 5

                cv2.circle(frame, (int(x), int(y)), size, color, -1)

        # 绘制躯干中线
        if len(tracker.angle_history) > 0:
            # 检查所有躯干关键点的置信度
            trunk_points = [5, 6, 11, 12]  # 肩膀和髋部点
            if all(keypoints[i][2] > confidence_threshold for i in trunk_points):
                center_up = (keypoints[5][:2] + keypoints[6][:2]) / 2
                center_down = (keypoints[11][:2] + keypoints[12][:2]) / 2

                # 检查中心点的有效性
                if (0 <= int(center_up[0]) < frame.shape[1] and
                        0 <= int(center_up[1]) < frame.shape[0] and
                        0 <= int(center_down[0]) < frame.shape[1] and
                        0 <= int(center_down[1]) < frame.shape[0]):
                    cv2.line(frame,
                             tuple(map(int, center_up)),
                             tuple(map(int, center_down)),
                             (0, 0, 255), 2)

        # 显示状态和角度
        state_color = {
            "Standing": (0, 255, 0),
            "Fallen": (0, 0, 255),
            "Error": (255, 0, 0),
            "Low Confidence": (128, 128, 128)
        }.get(state, (128, 128, 128))

        cv2.putText(frame, f"State: {state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)

        if len(tracker.angle_history) > 0:
            angle = tracker.angle_history[-1]
            cv2.putText(frame, f"Angle: {angle:.1f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    except Exception as e:
        print(f"可视化异常: {str(e)}")
        return frame


# 在主循环之前初始化多人跟踪器字典
pose_trackers = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    current_ids = set()  # 用于跟踪当前帧中的人

    for result in results:
        if result.keypoints is not None:
            # 处理每个检测到的人
            for person_id, keypoints in enumerate(result.keypoints.data):
                keypoints = keypoints.cpu().numpy()
                
                # 为新检测到的人创建跟踪器
                if person_id not in pose_trackers:
                    pose_trackers[person_id] = PoseTracker(history_size=10, time_window=2.0)
                
                current_ids.add(person_id)
                
                # 检测姿态状态
                state = detect_pose_state(keypoints, pose_trackers[person_id])
                
                # 可视化
                frame = visualize_skeleton(frame, keypoints, state, pose_trackers[person_id])
                
                # 添加人员ID标识
                if len(keypoints) > 0:
                    # 使用肩部位置作为ID显示位置
                    shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
                    if keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5:  # 确保肩部点置信度足够高
                        cv2.putText(frame, f"ID: {person_id}", 
                                  (int(shoulder_center[0]), int(shoulder_center[1] - 30)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # 在人物上方显示状态
                        status_y = int(shoulder_center[1] - 50)
                        state_color = {
                            "Standing": (0, 255, 0),
                            "Fallen": (0, 0, 255),
                            "Error": (255, 0, 0),
                            "Low Confidence": (128, 128, 128)
                        }.get(state, (128, 128, 128))
                        cv2.putText(frame, f"State: {state}", 
                                  (int(shoulder_center[0]), status_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)

    # 移除不再出现的人的跟踪器
    disappeared_ids = set(pose_trackers.keys()) - current_ids
    for disappeared_id in disappeared_ids:
        del pose_trackers[disappeared_id]

    # 显示当前检测到的人数
    cv2.putText(frame, f"Persons: {len(current_ids)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 显示推理结果
    cv2.imshow("Multi-Person Fall Detection", frame)

    # 如果按下'q'键，则退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

