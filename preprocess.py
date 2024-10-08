

import os
import json
import numpy as np


data_dir = 'G:/signData'
train_dir = os.path.join(data_dir,'train')
val_dir = os.path.join(data_dir,'vaild')
output_dir = os.path.join(data_dir,"nptxt_del")
train_landmark_dir = os.path.join(train_dir,'label','landmark')
train_morpheme_dir = os.path.join(train_dir,'label','morpheme')
val_landmark_dir = os.path.join(val_dir,'landmark')
val_morpheme_dir = os.path.join(val_dir,'morpheme')



def getoutputdir():
    return output_dir


# def load_data(person,word):
#     path=f"{output_dir}/{person}/{word}.npz"
#     data = np.load(path)
#     wordCoordL = data['wordCoordL']
#     wordCoordR = data['wordCoordR']
#     wordCoordP = data['wordCoordP']
#     label = data['label']
#

#     return wordCoordL, wordCoordR, wordCoordP, label
with open('wordtonum.json', 'r', encoding="UTF8") as json_file:
    words_dicts = json.load(json_file)

def load_data(file_name):
    path=f"{output_dir}/{file_name}"
    data = np.load(path)

    wordCoordL = data['wordCoordL']
    wordCoordR = data['wordCoordR']
    wordCoordP = data['wordCoordP']
    ans = data['label'][2]
    ans = ans.replace('\n', '')
    label = words_dicts[ans]
    return wordCoordL, wordCoordR, wordCoordP, label



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


