# YOLOv11 Multi-Person Fall Detection

This repository contains a Python script for multi-person fall detection using YOLOv11 and OpenCV.

## Introduction

This project uses YOLOv11 for pose estimation and OpenCV for video processing to detect and track multiple persons in a video. The goal is to identify states such as standing, sitting, and fallen using keypoint detection and state tracking.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/AIGlowHub/yolov11.git
    ```
2. Navigate to the project directory:
    ```sh
    cd yolov11
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Download or place your YOLOv11 model file (`yolo11m-pose.pt`) in the project directory.
2. Update the `main_mp4.py` script with the path to your video file and model file:
    ```python
    video_path = 'path/to/your/video.mp4'
    model = YOLO('path/to/your/yolo11m-pose.pt')
    ```
3. Run the script:
    ```sh
    python main_mp4.py
    ```

## Script Overview

The script performs the following steps:
1. Loads the YOLOv11 pose estimation model.
2. Captures video frames using OpenCV.
3. Detects keypoints and tracks each person in the video.
4. Determines the state (standing, sitting, fallen) of each person based on keypoint analysis.
5. Visualizes the detected keypoints and states on the video frames.
6. Displays the video with overlaid detection results.

## Model Principle

The YOLOv11 (You Only Look Once version 11) model is a state-of-the-art object detection model that achieves high accuracy and speed. The key principles of YOLOv11 used in this project are:

1. **Single-Stage Object Detection**: Unlike traditional object detection models that use a two-stage approach (region proposal followed by classification), YOLOv11 performs object detection in a single stage. This makes it faster and more efficient.

2. **Grid-Based Detection**: The input image is divided into a grid, and each grid cell is responsible for detecting objects within that cell. YOLOv11 predicts bounding boxes and confidence scores for each cell.

3. **Anchor Boxes**: YOLOv11 uses predefined anchor boxes of different shapes and sizes to predict bounding boxes more effectively. The model adjusts these anchor boxes to fit the detected objects.

4. **End-to-End Training**: YOLOv11 is trained end-to-end, meaning that the model learns to predict both bounding boxes and class probabilities simultaneously. This allows for better optimization and improved performance.

5. **Pose Estimation**: In this project, YOLOv11 is extended to include pose estimation. The model predicts keypoints for human body parts, which are used to determine the pose and state (standing, sitting, fallen) of each person.

6. **Multi-Person Tracking**: The script includes a pose tracker that maintains a history of keypoint positions, angles, and states for each detected person. This allows for accurate tracking and state determination over time.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project does not have a specific license. Please consult the repository owner for more information.

## Contact

For any inquiries or issues, please contact the repository owner at [AIGlowHub](https://github.com/AIGlowHub).

For more information, visit the [repository](https://github.com/AIGlowHub/yolov11).
