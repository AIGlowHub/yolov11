import cv2
import torch
import numpy as np
from time import time
from ultralytics import YOLO

# 加载本地YOLOv11-pose模型
model = YOLO('yolo11m-pose.pt')  # 替换为你的模型文件路径
model.conf = 0.8  # 设置置信度阈值
model.iou = 0.8  # 设置IOU阈值

# 加载本地视频
video_path = 'test2.mp4'  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

# 自动获取视频尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"视频信息：")
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
        self.state = "Standing"  # 添加状态初始化
        self.state_counts = {'Standing': 0, 'Sitting': 0, 'Fallen': 0}  # 添加状态计数器初始化
        self.fallen_frame_count = 0  # 添加连续摔倒帧计数器
        self.fallen_threshold = 3  # 需要连续3帧检测到摔倒才确认

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
    try:
        # 获取关键点坐标
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]

        # 计算上下身中心点
        center_up = (left_shoulder + right_shoulder) / 2
        center_down = (left_hip + right_hip) / 2
        knee_center = (left_knee + right_knee) / 2

        # 计算身体向量和腿部向量
        body_vector = center_down - center_up
        leg_vector = knee_center - center_down

        # 计算身体和腿部的夹角
        dot_product = np.dot(body_vector, leg_vector)
        body_norm = np.linalg.norm(body_vector)
        leg_norm = np.linalg.norm(leg_vector)
        
        if body_norm != 0 and leg_norm != 0:
            cos_angle = dot_product / (body_norm * leg_norm)
            body_leg_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        else:
            body_leg_angle = 0

        # 原有的角度计算
        right_angle_point = np.array([center_down[0], center_up[1]])
        a = abs(right_angle_point[0] - center_up[0])
        b = abs(center_down[1] - right_angle_point[1])
        angle = np.degrees(np.arctan(a / b)) if b != 0 else 90

        # 计算人体框的宽高比
        body_width = np.linalg.norm(left_shoulder - right_shoulder)
        body_height = np.linalg.norm(center_up - center_down)
        width_height_ratio = body_width / (body_height + 1e-6)

        # 计算膝盖到臀部的垂直距离和水平距离
        knee_hip_vertical = abs(knee_center[1] - center_down[1])
        knee_hip_horizontal = abs(knee_center[0] - center_down[0])
        
        # 计算膝盖角度（用于判断正面坐姿）
        knee_angle_ratio = knee_hip_horizontal / (knee_hip_vertical + 1e-6)

        # 更新跟踪器数据
        tracker.update_histories(center_up, body_height, angle)

        # 计算移动状态
        def calculate_relative_movement():
            if len(tracker.position_history) < 2:
                return float('inf'), 0

            # 计算身体特征
            def get_body_features(points):
                # 计算身体各部分的相对长度和角度
                shoulder_width = np.linalg.norm(points[5][:2] - points[6][:2])  # 肩宽
                hip_width = np.linalg.norm(points[11][:2] - points[12][:2])    # 臀宽
                body_height = np.linalg.norm(center_up - center_down)          # 躯干高度
                leg_length = np.linalg.norm(center_down - knee_center)         # 腿长
                
                # 计算身体比例（这些比例在人体移动时应该保持相对稳定）
                features = np.array([
                    shoulder_width / (body_height + 1e-6),
                    hip_width / (body_height + 1e-6),
                    leg_length / (body_height + 1e-6),
                    body_leg_angle,
                    angle
                ])
                return features

            # 获取当前帧的特征
            current_features = get_body_features(keypoints)
            
            # 计算特征变化
            movements = []
            for i in range(1, len(tracker.position_history)):
                # 计算特征的相对变化
                feature_change = np.abs(current_features - get_body_features(keypoints))
                # 使用特征变化的均值作为运动指标
                movement = np.mean(feature_change)
                movements.append(movement)

            # 计算平均变化和最近的最大变化
            avg_movement = np.mean(movements) if movements else float('inf')
            recent_max_movement = max(movements[-3:]) if len(movements) >= 3 else 0
            return avg_movement, recent_max_movement

        # 计算相对运动量
        current_movement, recent_max_movement = calculate_relative_movement()
        static_threshold = 0.1  # 静止状态阈值（基于特征变化）
        movement_threshold = 0.2  # 降低移动状态阈值，使其更容易检测到移动

        # 判断移动状态
        is_static = current_movement < static_threshold
        is_moving = recent_max_movement > movement_threshold

        current_fallen = (
            angle > 60 or  # 角度大于60度
            center_down[1] <= center_up[1] or  # 上下身中心点位置异常
            width_height_ratio > 5/3  # 宽高比异常
        )

        # 更新连续帧计数
        if current_fallen:  # 移除移动状态判断，使其更容易触发
            tracker.fallen_frame_count += 1
        else:
            tracker.fallen_frame_count = 0

        # 降低连续帧判断阈值
        tracker.fallen_threshold = 8  # 只需要2帧就确认摔倒状态

        # 只有当连续多帧检测到摔倒时才确认是摔倒状态
        is_fallen = tracker.fallen_frame_count >= tracker.fallen_threshold

        # 修改坐姿判断条件，增加静止状态判断
        is_sitting = (
            is_static and  # 必须是静止状态
            current_movement < static_threshold and  # 确保特征变化很小
            (20 <= body_leg_angle <= 175) and  # 身体和腿部夹角范围
            (
                (10 <= angle <= 120) or  # 躯干倾斜角度范围
                knee_angle_ratio > 0.2 or  
                np.linalg.norm(center_down - knee_center) < body_height * 0.9
            ) and
            (
                center_down[1] > center_up[1] * 0.95 or
                knee_angle_ratio > 0.3
            ) and
            not is_fallen  # 确保不是跌倒状态
        )

        # 更新状态计数
        if is_fallen:  # 移除移动状态判断
            if tracker.state_counts['Fallen'] > 0:
                tracker.state_counts['Fallen'] += 2
            else:
                tracker.state_counts['Fallen'] += 1
            tracker.state_counts['Standing'] = 0
            tracker.state_counts['Sitting'] = 0
        elif is_sitting and is_static:  # 只有在静止状态下才更新坐姿计数
            tracker.state_counts['Sitting'] += 1
            tracker.state_counts['Standing'] = 0
            tracker.state_counts['Fallen'] = 0
        else:
            tracker.state_counts['Standing'] += 1
            tracker.state_counts['Sitting'] = 0
            tracker.state_counts['Fallen'] = 0

        # 状态判断阈值
        fallen_threshold = 1  # 降低摔倒判断所需帧数
        sitting_threshold = 5  # 保持坐姿判断帧数不变
        standing_threshold = 25  # 保持站立判断帧数不变

        # 最终状态判断
        if tracker.state_counts['Fallen'] >= fallen_threshold:  # 移除移动状态判断
            if tracker.state != "Fallen":
                tracker.fall_start_time = time()
            tracker.stable_count = 0
            tracker.state = "Fallen"
        elif is_static and tracker.state_counts['Sitting'] >= sitting_threshold:
            tracker.state = "Sitting"
            tracker.stable_count = 0
        else:
            tracker.stable_count += 1
            if tracker.stable_count > standing_threshold:
                tracker.state = "Standing"
                tracker.fall_start_time = None

        return tracker.state

    except Exception as e:
        print(f"检测异常: {str(e)}")
        return "Error"


def visualize_skeleton(frame, keypoints, state, tracker):
    try:
        # 提高置信度阈值
        confidence_threshold = 0.9  # 提高置信度阈值

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

        # cv2.putText(frame, f"State: {state}", (20, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)

        if len(tracker.angle_history) > 0:
            angle = tracker.angle_history[-1]
            # cv2.putText(frame, f"Angle: {angle:.1f}", (20, 80),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

    # 统计各种状态的人数
    standing_count = sum(1 for pid in current_ids if pose_trackers[pid].state == "Standing")
    sitting_count = sum(1 for pid in current_ids if pose_trackers[pid].state == "Sitting")
    fallen_count = sum(1 for pid in current_ids if pose_trackers[pid].state == "Fallen")
    total_count = len(current_ids)

    # 显示状态统计信息
    cv2.putText(frame, f"Total: {total_count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Standing: {standing_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Sitting: {sitting_count}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    cv2.putText(frame, f"Fallen: {fallen_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示推理结果
    cv2.imshow("Multi-Person Fall Detection", frame)

    # 如果按下'q'键，则退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

