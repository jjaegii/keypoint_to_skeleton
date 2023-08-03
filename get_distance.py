import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

keypoints_video1 = []
keypoints_video2 = []

cap1 = cv2.VideoCapture('video1.mp4')
cap2 = cv2.VideoCapture('stand.mp4')

# 두 비디오 중 더 짧은 길이를 구합니다.
length = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for _ in range(length):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        image1.flags.writeable = False
        image2.flags.writeable = False

        results1 = pose.process(image1)
        results2 = pose.process(image2)

        # 양쪽 비디오에서 키포인트를 찾을 수 없는 경우, (0,0,0)으로 채웁니다.
        keypoints1 = [(landmark.x, landmark.y, landmark.z) for landmark in results1.pose_landmarks.landmark] if results1.pose_landmarks is not None else [(0, 0, 0)]
        keypoints2 = [(landmark.x, landmark.y, landmark.z) for landmark in results2.pose_landmarks.landmark] if results2.pose_landmarks is not None else [(0, 0, 0)]

        keypoints_video1.append(keypoints1)
        keypoints_video2.append(keypoints2)

cap1.release()
cap2.release()

# 두 비디오에서 추출된 키포인트의 수가 동일하다고 가정하고 계산합니다.
distances = []
for keypoints1, keypoints2 in zip(keypoints_video1, keypoints_video2):
    for point1, point2 in zip(keypoints1, keypoints2):
        dist = distance.euclidean(point1, point2)
        distances.append(dist)

avg_distance = np.mean(distances)

# 여기서는 (0, 0, 0)과 (1, 1, 1) 사이의 거리를 최대 거리로 가정합니다.
# 이는 Mediapipe 키포인트의 x, y, z 좌표가 일반적으로 [0, 1] 범위 내에 있기 때문입니다.
max_distance = distance.euclidean((0, 0, 0), (1, 1, 1))

# 유사성 점수는 1에서 (평균 거리 / 최대 거리)를 뺀 값입니다. 이 값이 1에 가까울수록 키포인트가 유사하다는 것을 나타냅니다.
similarity_score = 1 - (avg_distance / max_distance)

print("Similarity score between the two videos:", similarity_score * 100, "%")