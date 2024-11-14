import cv2

video_path = r"A:\DeepFakeDetectionTool\Deepfake_Detection_Tool\datasets\Celeb-DF-v2\Celeb-synthesis\id0_id9_0001.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
else:
    print("Video opened successfully")
