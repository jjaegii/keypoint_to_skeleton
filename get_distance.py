import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

keypoints_video1 = []
keypoints_video2 = []

cap1 = cv2.VideoCapture('video1.mp4')
cap2 = cv2.VideoCapture('video1.mp4')
# 1. 키포인트 뽑아서 비교해보깅
# 2. 소수점 반올림 후 측정

# 두 비디오 중 더 짧은 길이를 구합니다.
length = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

prev_keypoints1 = [(0, 0, 0)] * 33
prev_keypoints2 = [(0, 0, 0)] * 33

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

        # 검출되는 키포인트 부분이 50% 아래면 직전 프레임의 키포인트로 처리
        for i in range(0, 33):
            if results1.pose_landmarks.landmark[i].visibility < 0.5:
                results1.pose_landmarks.landmark[i].x, results1.pose_landmarks.landmark[i].y, results1.pose_landmarks.landmark[i].z = prev_keypoints1[i]
            else:
                prev_keypoints1[i] = (results1.pose_landmarks.landmark[i].x, results1.pose_landmarks.landmark[i].y, results1.pose_landmarks.landmark[i].z)
            
            if results2.pose_landmarks.landmark[i].visibility < 0.5:
                results2.pose_landmarks.landmark[i].x, results2.pose_landmarks.landmark[i].y, results2.pose_landmarks.landmark[i].z = prev_keypoints2[i]
            else:
                prev_keypoints2[i] = (results2.pose_landmarks.landmark[i].x, results2.pose_landmarks.landmark[i].y, results2.pose_landmarks.landmark[i].z)

        keypoints1 = [(np.round(landmark.x, 2), np.round(landmark.y, 2), np.round(landmark.z, 2)) for landmark in results1.pose_landmarks.landmark]
        keypoints_video1.append(keypoints1)

        keypoints2 = [(np.round(landmark.x, 2), np.round(landmark.y, 2), np.round(landmark.z, 2)) for landmark in results2.pose_landmarks.landmark]
        keypoints_video2.append(keypoints2)

cap1.release()
cap2.release()

import pickle

with open('keypoints_video1.pkl', 'wb') as f:
    pickle.dump(keypoints_video1, f)

with open('keypoints_video2.pkl', 'wb') as f:
    pickle.dump(keypoints_video2, f)


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
