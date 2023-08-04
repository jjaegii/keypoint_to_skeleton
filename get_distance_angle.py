import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    # Calculate the angle between three points a, b, and c
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle

keypoints_video1 = []
keypoints_video2 = []

cap1 = cv2.VideoCapture('video1.mp4')
cap2 = cv2.VideoCapture('stand.mp4')

# Get the frame rate of video1.mp4
frame_rate = cap1.get(cv2.CAP_PROP_FPS)
# Calculate the starting frame to crop 2 seconds (assuming 2 seconds is the desired duration to crop)
start_frame = int(frame_rate * 3.3)

# Set the starting frame for 'video1.mp4' capture
cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

length = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for _ in range(length):
        ret1, frame1 = cap1.read()
        frame1 = cv2.flip(frame1, 1)
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        image1.flags.writeable = False
        image2.flags.writeable = False

        results1 = pose.process(image1)
        results2 = pose.process(image2)

        keypoints1 = [(landmark.x, landmark.y, landmark.z) for landmark in results1.pose_landmarks.landmark] if results1.pose_landmarks is not None else [(0, 0, 0)]
        keypoints2 = [(landmark.x, landmark.y, landmark.z) for landmark in results2.pose_landmarks.landmark] if results2.pose_landmarks is not None else [(0, 0, 0)]

        keypoints_video1.append(keypoints1)
        keypoints_video2.append(keypoints2)

cap1.release()
cap2.release()

distances = []
angles = []
for keypoints1, keypoints2 in zip(keypoints_video1, keypoints_video2):
    for point1, point2 in zip(keypoints1, keypoints2):
        dist = distance.euclidean(point1, point2)
        distances.append(dist)

    # Calculate the angles between the arms and legs and add them to the angles list
    left_shoulder = keypoints1[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = keypoints1[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = keypoints1[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angles.append(left_angle)

    right_shoulder = keypoints1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = keypoints1[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = keypoints1[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    angles.append(right_angle)

    left_hip = keypoints1[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = keypoints1[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = keypoints1[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    angles.append(left_leg_angle)

    right_hip = keypoints1[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = keypoints1[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = keypoints1[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
    angles.append(right_leg_angle)

avg_distance = np.mean(distances)
avg_angle = np.mean(angles)

max_distance = distance.euclidean((0, 0, 0), (1, 1, 1))
max_angle = 180  # Maximum possible angle value

# Calculate the similarity score with 60% weight for avg_angle and 40% weight for avg_distance
similarity_score = 0.3 * (1 - (avg_angle / max_angle)) + 0.7 * (1 - (avg_distance / max_distance))

print("Similarity score between the two videos:", similarity_score * 100, "%")
