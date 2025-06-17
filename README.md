# School Video Analysis

A professional video analysis tool for pose estimation in school environments, built with Streamlit and OpenCV. This application enables users to upload classroom videos and automatically process them to detect hand-raising gestures frame-by-frame using a pretrained YOLOv11 pose estimation model from Ultralytics.

## Features

- **Video Upload:** Supports MP4 and AVI formats for easy classroom video uploads.
- **Pose Estimation:** Processes each frame to detect hand-raising gestures using a pretrained YOLOv11 pose model.
- **YOLOv11 Integration:** Utilizes pretrained YOLOv11 models from Ultralytics for robust pose and face detection.
- **Real-Time Visualization:** Displays processed frames in real-time during analysis with bounding boxes, keypoints, and hand-raising annotations.
- **Processed Video Output:** Saves and allows playback of the processed video with pose annotations overlayed.

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- OpenCV
- Ultralytics YOLO (for YOLOv11 models)
- Other dependencies as required by your models (see below)

### Installation

It is recommended to use a dedicated conda environment for this project. Below are the steps to set up the environment with Python 3.12:

```bash
conda create -n pose-env python=3.12 -y
conda activate pose-env
```

Clone this repository:

```bash
git clone <your-repo-url>
cd pose
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ultralytics and OpenCV for pose estimation:

```bash
pip install ultralytics
pip install opencv-python
```

Place your trained models in the `models/` directory.

## Usage

Start the Streamlit app:

```bash
streamlit run streamlit.py
```
![Result](pose.gif)
- Upload a classroom video (MP4 or AVI).
- Click **Start processing** to analyze the video for hand-raising gestures.
- View real-time results and download the processed video with pose annotations.

## Project Structure

```
.
├── pose.py                  # Pose estimation logic
├── streamlit.py             # Streamlit web application
├── models/                  # Pre-trained model files (including YOLOv11 from Ultralytics)
├── videos/                  # Sample videos
└── ...
```

## Model Files

Place your PyTorch models in the `models/` directory.

Example model files:
- `yolo11n-pose.pt` (YOLOv11 pose estimation model from Ultralytics)

## Using Pretrained YOLOv11 from Ultralytics

This project leverages pretrained YOLOv11 models from Ultralytics for pose estimation.
The `yolo11n-pose.pt` model is used to detect keypoints (e.g., shoulders, elbows, wrists) and identify hand-raising gestures based on their relative positions.
You can download official YOLOv11 models or train your own using the Ultralytics framework.
To use a pretrained model, place the `.pt` file in the `models/` directory and ensure your code loads it with the Ultralytics API:

```python
from ultralytics import YOLO
model = YOLO('models/yolo11n-pose.pt')
results = model(frame)
```

For more details, see the [Ultralytics YOLO documentation](https://docs.ultralytics.com/).

## Pose Estimation Details

The pose estimation logic detects hand-raising by analyzing the positions of shoulders, elbows, and wrists.
A hand is considered raised if the wrist is above the elbow and the elbow is above the shoulder in the y-coordinate.
Keypoints are drawn as circles, and lines connect shoulders to elbows and elbows to wrists for visualization.
Bounding boxes and text annotations ("Raises his hand") are added to frames where hand-raising is detected.

Example usage in code:

```python
from ultralytics import YOLO
model = YOLO('models/yolo11n-pose.pt')
results = model(frame, conf=0.75)
for result in results:
    keypoints = result.keypoints.xy
    boxes = result.boxes.xyxy
    # Process keypoints and boxes for hand-raising detection
```

## License

This project is intended for educational and research purposes. Please review and comply with your institution's data privacy and ethical guidelines when using classroom videos.
