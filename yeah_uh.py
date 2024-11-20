from tensorflow.math import top_k
import tensorflow as tf
import SLR_model_CNN_GRU as SLR
from collections import Counter

VOTE_WDW_SIZE = 5
THRESHOLD = 0.6
last_yeh = -1
vote_counter = Counter()
vote_wdw = [0]

for i in range(110):
    # just pretend this is the model outputting result
    prediction = tf.random.uniform((4,))

    pred_res = SLR.decode_onehot2d(prediction).numpy()
    top2 = top_k(prediction, k=2)
    top2_np = top2.values.numpy()

    conf_weight = top2_np[0] - top2_np[1]

    vote_wdw.append({pred_res : conf_weight})
     
    if len(vote_wdw) > VOTE_WDW_SIZE:
        vote_wdw.pop(0)
        vote_counter.clear()
        for elem in vote_wdw: # 0 1 2 3
            vote_counter.update(elem)
        top = vote_counter.most_common(1)[0]
        if top[1] > THRESHOLD:
            if top[0] != last_yeh:
                print('yeh:', top[0])
                last_yeh = top[0]
            else:
                print('nuhuh overlap: ', last_yeh, top[0])
        print('erm:', top[1]) 