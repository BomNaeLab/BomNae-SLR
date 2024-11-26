import numpy as np
import json
from collections import Counter
from tensorflow.math import top_k
import SLR_model as SLR
import tensorflow as tf

# 다수결 관련 설정
VOTE_WDW_SIZE = 5
THRESHOLD = 0.6
last_output = -1
vote_wdw = []
vote_counter = Counter()

# 글로벌 모델 초기화
model = None

def initialize_model(model_path="11-20-16-30epochs-5times_final.keras"):
    """
    모델 초기화 함수. 처음 로드 시 전역 변수에 저장.
    """
    global model
    if model is None:
        print("Loading model...")
        model = SLR.load_model(model_path)
    return model

def serialize(vid, stride): #, is_pose=False):
    window = vid[::stride]
    # if is_pose:
    #     window = np.swapaxes(window,-1, -2)
    return np.array(window)

def vote_system(l, r, p, model_path="11-20-16-30epochs-5times_final.keras"):
    """
    63프레임 데이터를 받아 모델 예측을 실행하고, 다수결로 결과를 도출하는 함수.
    """
    global last_output, vote_wdw, vote_counter

    # 모델 초기화 (최초 1회만 실행)
    model = initialize_model(model_path)
    ex_l=tf.expand_dims(serialize(l,1),axis=0)
    ex_r=tf.expand_dims(serialize(r,1),axis=0)
    ex_p=tf.expand_dims(serialize(p,2),axis=0)
    # 데이터 직렬화
    x_test = (ex_l,ex_r,ex_p)
    print(f'l={x_test[0].shape} r={x_test[1].shape} p={x_test[2].shape}')
    
    
    if len(l)==0:
        return ""
    # 모델 예측
    prediction = model.predict(x_test, verbose=0)
    pred_res = SLR.decode_onehot2d(prediction).numpy()
    top2 = top_k(prediction, k=2)
    top2_np = top2.values.numpy()
    conf_weight = top2_np[0][0] - top2_np[0][1]
    pred_res_tuple = tuple(pred_res.tolist()) 
    vote_wdw.append({pred_res_tuple: conf_weight})

    # 다수결 창 크기 유지
    if len(vote_wdw) > VOTE_WDW_SIZE:
        vote_wdw.pop(0)

    # 다수결 계산
    vote_counter.clear()
    for elem in vote_wdw:
        vote_counter.update(elem)

    top = vote_counter.most_common(1)[0]
    if top[1] > THRESHOLD:
        if top[0] != last_output:
            last_output = top[0]
            output_num = top[0]

            # JSON에서 번호를 단어로 변환
            with open('wordtonum_lite_old.json', 'r', encoding="UTF8") as json_file:
                data = json.load(json_file)
                result = next((key for key, value in data.items() if value == output_num), None)
        else:
            result = "not"
    else:
        result = "not"  # Threshold 미만이면 빈 문자열 반환

    return result