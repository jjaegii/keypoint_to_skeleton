import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_skeleton(frame, landmarks):
    # Draw the skeleton overlay on the frame
    mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

def main():
    cap_video2 = cv2.VideoCapture('video1.mp4')
    
    cap_stand = cv2.VideoCapture('stand.mp4')

    # Get the frame rate of video1.mp4
    frame_rate = cap_video2.get(cv2.CAP_PROP_FPS)
    # Calculate the starting frame to crop 2 seconds (assuming 2 seconds is the desired duration to crop)
    start_frame = int(frame_rate * 3.3)

    # Set the starting frame for 'video1.mp4' capture
    cap_video2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret1, frame_video2 = cap_video2.read()
            ret2, frame_stand = cap_stand.read()

            frame_video2 = cv2.flip(frame_video2, 1)

            if not ret1 or not ret2:
                break

            image_video2 = cv2.cvtColor(frame_video2, cv2.COLOR_BGR2RGB)
            image_stand = cv2.cvtColor(frame_stand, cv2.COLOR_BGR2RGB)

            image_video2.flags.writeable = False
            image_stand.flags.writeable = False

            results_video2 = pose.process(image_video2)
            results_stand = pose.process(image_stand)

            if results_video2.pose_landmarks:
                # Draw the skeleton for 'video2.mp4'
                frame_video2_skeleton = np.zeros_like(frame_video2)
                draw_skeleton(frame_video2_skeleton, results_video2.pose_landmarks)

                cv2.imshow('Skeleton for video2.mp4', frame_video2_skeleton)

            if results_stand.pose_landmarks:
                # Draw the skeleton for 'stand.mp4'
                frame_stand_skeleton = np.zeros_like(frame_stand)
                draw_skeleton(frame_stand_skeleton, results_stand.pose_landmarks)

                cv2.imshow('Skeleton for stand.mp4', frame_stand_skeleton)

            if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' key to exit
                break

    cap_video2.release()
    cap_stand.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
