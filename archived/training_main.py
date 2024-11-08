
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
from tensorflow import keras
import time
import json
import preprocess
# import preprocess as prep
import archived.SLR_model_GRU_legacy as SLR_model_GRU_legacy
import numpy as np

load_size = 3000 # number of data to be loaded at once
epochs = 50
run_time=2
batch_size = 16
save_dir = "saves_GRU"
load_dir = "saves_GRU"
model = SLR_model_GRU_legacy.get_model()

 # reload model file
end_file=preprocess.getoutputdir()

save_suffix = time.strftime("%m-%d-%H", time.localtime(time.time()))
ckpt_name=save_suffix+"-"+str(epochs)+"epochs-"+str(run_time)+"times"

check_path = os.path.join(save_dir,'ckpt',ckpt_name)
hist_path = os.path.join(save_dir, "hist",ckpt_name+".json")

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
filepath=f'{check_path}.keras',
monitor='categorical_accuracy',
mode='max',
save_freq='epoch',
save_best_only=True)


start_person=1
start_count=1
start_word=" "
with open(os.path.join('logs',ckpt_name+'.txt'), 'a') as logs:
    for i in range(1,17):
        l_raws=[]
        r_raws=[]
        p_raws=[]
        y_raws=[]
        loss_weights_raws=[]
        if start_person>i:
            continue
        for k in range(1,run_time+1):
            if start_count>k:
                continue
            elif start_count==k:
                start_count=0
            for j in sorted(os.listdir(os.path.join(preprocess.getoutputdir(),str(i)))):
                if len(l_raws)==0:
                    start_word = j
                else:
                    end_word = j
                l_raw, r_raw, p_raw, y_raw, loss_weights_raw = preprocess.load_data(f"{i}/{j}")
                l_raws.append(l_raw)
                r_raws.append(r_raw)
                p_raws.append(p_raw)
                y_raws.append(y_raw)
                loss_weights_raws.append(loss_weights_raw)

                if len(l_raws)>=load_size:
                    
                    logs.write(f'{ time.strftime("%H-%M-%S", time.localtime(time.time()))}:{k}) person:{i} : {start_word} ~ {end_word}\n')  # 한 줄 쓰기
                    l_train, each = SLR_model_GRU_legacy.serialize(l_raws)
                    r_train, each = SLR_model_GRU_legacy.serialize(r_raws)
                    p_train, each, sample_weights = SLR_model_GRU_legacy.serialize(p_raws, stride=2, loss_weights_list=loss_weights_raws)
                    x_train = (l_train, r_train, p_train)
                    
                    y_train = np.repeat(y_raws, each)
                    y_train = SLR_model_GRU_legacy.encode_onehot2d(y_train)
                    
                    dataset = SLR_model_GRU_legacy.convert_to_dataset(x_train, y_train, batch_size, sample_weights)
                    hist = model.fit(dataset, epochs=epochs, callbacks=[model_checkpoint_callback])

                    with open(hist_path, 'w') as file:
                        json.dump(hist.history, file)
                    l_raws.clear()
                    r_raws.clear()
                    p_raws.clear()
                    y_raws.clear()
                    loss_weights_raws.clear()


