

import os
import json
import numpy as np

data_dir = 'G:/signData'

train_dir = os.path.join(data_dir,'train')
val_dir = os.path.join(data_dir,'valid')

output_dir = os.path.join(data_dir,'nptxt')
val_output_dir = os.path.join(data_dir,"nptxt_val")
weight_dir = os.path.join(data_dir,'weights')

train_landmark_dir = os.path.join(train_dir,'label','landmark')
train_morpheme_dir = os.path.join(train_dir,'label','morpheme')

val_landmark_dir = os.path.join(val_dir,'landmark')
val_morpheme_dir = os.path.join(val_dir,'morpheme')


def getoutputdir(type="train"):
    return val_output_dir if type=='val' else output_dir

with open('wordtonum.json', 'r', encoding="UTF8") as json_file:
    words_dicts = json.load(json_file)

def load_data(file_name,type="train"):
    path=f"{getoutputdir(type)}/{file_name}"
    weight_path = f"{weight_dir}/{file_name}"
    data = np.load(path)
    weight_data = np.load(weight_path)
    #좌표값 로드
    wordCoordL = data['wordCoordL']
    wordCoordR = data['wordCoordR']
    wordCoordP = data['wordCoordP']
    #단어 뜻 호출
    ans = data['label'][0]
    # ans = ans.replace('\n', '')
    #해당 단어의 value 호출
    label = words_dicts[ans]
    weight = weight_data['weight']
    return wordCoordL, wordCoordR, wordCoordP, label ,weight



def load_word(person, start, num):
    coordLs = []
    coordRs = []
    coordPs = []
    labels = []
    checks = []
    for WNum in range(start, start + num):
        wordCoordL, wordCoordR, wordCoordP, label = load_data(person, WNum)
        coordLs.append(wordCoordL)
        coordRs.append(wordCoordR)
        coordPs.append(wordCoordP)
        labels.append(int(label[0]))
        checks.append((label[1:3]))

    return coordLs, coordRs, coordPs, labels, checks


