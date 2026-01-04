"""
Pose worker (V14).

- Decoder: OpenCV
- Pose: MediaPipe
- Output: per-frame pose landmarks
- No frame drop
- No smoothing
"""

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


def run_pose(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)

    pose_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = [
                (lm.x, lm.y, lm.z)
                for lm in result.pose_landmarks.landmark
            ]
        else:
            landmarks = None

        pose_frames.append({
            "frame": frame_idx,
            "landmarks": landmarks
        })

        frame_idx += 1

    cap.release()
    pose.close()

    return pose_frames
