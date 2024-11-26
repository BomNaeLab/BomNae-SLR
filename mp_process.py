import cv2
import mediapipe as mp

# MediaPipe Hands, Pose 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

class MediaPipeProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_with_mediapipe(self, frame_input):
        """
        MediaPipe를 사용하여 손과 포즈 랜드마크를 추출하는 함수.
        
        :param frame_input: 입력 영상 프레임 (numpy 배열)
        :return: 랜드마크 좌표를 Python 리스트로 반환
        """
        frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)

        # 손 랜드마크 추출
        left_hand_landmarks = None
        right_hand_landmarks = None
        results_hands = self.hands.process(frame)
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                hand_landmarks_list = [
                    [landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark
                ]

                # 왼손 또는 오른손 구분
                if handedness.classification[0].label == 'Left':
                    left_hand_landmarks = hand_landmarks_list
                elif handedness.classification[0].label == 'Right':
                    right_hand_landmarks = hand_landmarks_list

        # 손이 하나만 인식되면, 다른 손을 0으로 채움
        if left_hand_landmarks is None:
            left_hand_landmarks = [[0, 0, 0]] * 21  # 왼손에는 21개의 랜드마크가 있음
        if right_hand_landmarks is None:
            right_hand_landmarks = [[0, 0, 0]] * 21  # 오른손에도 21개의 랜드마크가 있음

        # 포즈 랜드마크 추출
        pose_landmarks = []
        results_pose = self.pose.process(frame)
        if results_pose.pose_landmarks:
            pose_landmarks = [
                [landmark.x, landmark.y, landmark.z] for landmark in results_pose.pose_landmarks.landmark
            ]
        else:
            # 포즈 랜드마크가 없을 경우, 33개의 랜드마크를 0으로 채움
            pose_landmarks = [[0, 0, 0]] * 33

        # 원하는 출력 형식으로 변환
        pose_keypoints_3d = [coord for point in pose_landmarks for coord in point]  # 포즈 랜드마크 3D 리스트
        hand_left_keypoints_3d = [coord for point in left_hand_landmarks for coord in point]  # 왼손 랜드마크 3D 리스트
        hand_right_keypoints_3d = [coord for point in right_hand_landmarks for coord in point]  # 오른손 랜드마크 3D 리스트
        
        return {
            "pose_keypoints_3d": pose_keypoints_3d,
            "hand_left_keypoints_3d": hand_left_keypoints_3d,
            "hand_right_keypoints_3d": hand_right_keypoints_3d
        }

    def close(self):
        """MediaPipe 자원을 해제합니다."""
        self.hands.close()
        self.pose.close()