import cv2
import mediapipe as mp

# MediaPipe Hands, FaceMesh, Pose 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # 동영상 스트림을 처리하므로 False로 설정
    max_num_hands=2,          # 추적할 손의 최대 수
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,       # 복잡도 설정 (1이 기본)
    smooth_landmarks=True,    # 랜드마크를 부드럽게 처리
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_with_mediapipe(frame_input):
    frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (1080, 1080))
    """
    MediaPipe를 사용하여 비디오 프레임에서 랜드마크를 추출하는 함수.
    
    :param frame: 입력 영상 프레임 (numpy 배열)
    :return: 랜드마크 좌표 목록
    """
    landmarks = []

    # 손 랜드마크 추출
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))

    # 얼굴 랜드마크 추출 (FaceMesh)
    # face_results = face_mesh.process(frame)
    # if face_results.multi_face_landmarks:
    #     for face_landmarks in face_results.multi_face_landmarks:
    #         for landmark in face_landmarks.landmark:
    #             landmarks.append((landmark.x, landmark.y))

    # Pose 랜드마크 추출
    pose_results = pose.process(frame)
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

    return landmarks

def process_with_mediapipe_apart(frame_input):
    """
    MediaPipe를 사용하여 비디오 프레임에서 손(왼손, 오른손)과 포즈 랜드마크를 추출하는 함수.
    
    :param frame: 입력 영상 프레임 (numpy 배열)
    :return: 왼손 랜드마크, 오른손 랜드마크, 포즈 랜드마크를 포함한 NumPy 배열
    """
    left_hand_landmarks = []
    right_hand_landmarks = []
    pose_landmarks = []
    
    frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (1080, 1080))
    # 손 랜드마크 추출
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks_in_frame, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_landmarks_single = []
            for landmark in hand_landmarks_in_frame.landmark:
                hand_landmarks_single.append((landmark.x, landmark.y, landmark.z))

            # 왼손과 오른손 구분
            if handedness.classification[0].label == 'Left':
                left_hand_landmarks.append(hand_landmarks_single)
            elif handedness.classification[0].label == 'Right':
                right_hand_landmarks.append(hand_landmarks_single)

    # Pose 랜드마크 추출
    pose_results = pose.process(frame)
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            pose_landmarks.append((landmark.x, landmark.y, landmark.z))

    # NumPy 배열로 변환
    # left_hand_landmarks = np.array(left_hand_landmarks) if left_hand_landmarks else np.array([])
    # right_hand_landmarks = np.array(right_hand_landmarks) if right_hand_landmarks else np.array([])
    # pose_landmarks = np.array(pose_landmarks) if pose_landmarks else np.array([])

    return left_hand_landmarks, right_hand_landmarks, pose_landmarks