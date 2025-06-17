"""
pip install ultralytics
pip install opencv-python
"""
import cv2
from ultralytics import YOLO

# Load YOLO pose model
model = YOLO("models/yolo11s-pose.pt")
def process_pose_frame(frame):
    # 1) Run pose estimation
    results = model(frame, conf=0.75)

    # 2)
    for result in results:
        keypoints = result.keypoints.xy  # Extract keypoints
        boxes = result.boxes.xyxy  # Extract bounding boxes 
    
    for kp, box in zip(keypoints, boxes):
        kp = kp.cpu().numpy()
        box = box.cpu().numpy()

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract relevant keypoints
        left_shoulder = kp[6]
        right_shoulder = kp[7]
        left_elbow = kp[8]
        right_elbow = kp[9]
        left_wrist = kp[10]
        right_wrist = kp[11]

        # Check if wrists are in top-left of the image
        if not (left_wrist[0] < frame.shape[1] * 0.2 and left_wrist[1] < frame.shape[0] * 0.2) and not (right_wrist[0] < frame.shape[1] * 0.2 and right_wrist[1] < frame.shape[0] * 0.2):
            # Draw keypoints
            for idx in [6, 7, 8, 9, 10, 11]:
                x, y = map(int, kp[idx])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Draw lines connecting keypoints
            cv2.line(frame, tuple(map(int, left_shoulder)), tuple(map(int, left_elbow)), (255, 0, 0), 2)
            cv2.line(frame, tuple(map(int, left_elbow)), tuple(map(int, left_wrist)), (255, 0, 0), 2)
            cv2.line(frame, tuple(map(int, right_shoulder)), tuple(map(int, right_elbow)), (255, 0, 0), 2)
            cv2.line(frame, tuple(map(int, right_elbow)), tuple(map(int, right_wrist)), (255, 0, 0), 2)

            # Check hand raise condition (wrist > elbow > shoulder)
            if left_wrist[1] < left_elbow[1] and left_elbow[1] < left_shoulder[1]:
                cv2.putText(frame, "Raises his hand", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if right_wrist[1] < right_elbow[1] and right_elbow[1] < right_shoulder[1]:
                cv2.putText(frame, "Raises his hand", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame       


def main(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"فشل في فتح الفيديو: {video_path}")
        return

    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Estimation", 1500, 1200)

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        processed_frame = process_pose_frame(frame)

        cv2.imshow("Pose Estimation", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "videos/video4.mp4" 
    main(video_path)