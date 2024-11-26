import numpy as np

def collect_data(left_hand_data, right_hand_data, pose_data):
    """
    deque 데이터를 하나의 일관된 입력 형식으로 변환.
    """
    if len(left_hand_data) < 63 or len(right_hand_data) < 63 or len(pose_data) < 63:
        raise ValueError("Not enough data in the deque to form a complete input.")

    # deque 데이터를 numpy 배열로 변환
    l = np.array(list(left_hand_data))
    r = np.array(list(right_hand_data))
    p = np.array(list(pose_data))

    return l, r, p