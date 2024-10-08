train_dir = 'G:/signData'
#train_dir = '../data/signData'
output_dir = f"{train_dir}/nptxt"
json_folder_path = f'{train_dir}/train/label/landmark'
morpheme_path = f'{train_dir}/train/label/morpheme'

import os
import json
import numpy as np

def getoutputdir():
    return output_dir


start_person="17"
for person in os.listdir(json_folder_path):
    if int(person) < int(start_person):
        continue
    person_output_path = os.path.join(output_dir, str(int(person)))
    os.makedirs(person_output_path, exist_ok=True)
    word_list=[]
    for word_coords,word_morpheme in zip(os.listdir(os.path.join(json_folder_path, person)),os.listdir(os.path.join(morpheme_path, person))):
        if "F" in word_morpheme:
            file_path = os.path.join(morpheme_path, person, word_morpheme)
            morpheme_file_path = os.path.join(morpheme_path, person, word_morpheme)
            with open(morpheme_file_path, 'r', encoding="UTF8") as morpheme_file:
                data = json.load(morpheme_file)
                try:
                    name = data['data'][0]['attributes'][0]['name']
                    # word_list.append({data['metaData']['name'][7:15]:name})
                except IndexError as e:
                    name=False
                    # word_list.append({data['metaData']['name'][7:15]:None})
                    print(f"Error reading {morpheme_file_path}: {e}")
                    continue
        if "F" in word_coords and name is not False:
            wordCoordL = np.empty((0, 4, 5, 3))
            wordCoordR = np.empty((0, 4, 5, 3))
            wordCoordP = np.empty((0, 3, 10))
            word_num=int(word_coords[11:15])
            for frame in os.listdir(os.path.join(json_folder_path, person, word_coords)):
                file_path = os.path.join(json_folder_path, person, word_coords, frame)
                try:
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        lh_points = data['people']['hand_left_keypoints_3d']
                        rh_points = data['people']['hand_right_keypoints_3d']
                        p_points = data['people']['pose_keypoints_3d']

                        preFrameCoordP = np.array([[(960 * p_points[i] + 960 - 420) / (1500 - 420),
                                                    (1080 * p_points[i + 1] + 540) / 1080,
                                                    (p_points[32 + 2] - p_points[i + 2]) / 10]
                                                   for i in range(0, len(p_points), 4)], dtype=np.float32)

                        preFrameCoordL = np.array([[(960 * lh_points[i] + 960 - 420) / (1500 - 420),
                                                    (1080 * lh_points[i + 1] + 540) / 1080,
                                                    (lh_points[2] - lh_points[i + 2]) / 10]
                                                   for i in range(4, len(lh_points), 4)], dtype=np.float32)

                        preFrameCoordR = np.array([[(960 * rh_points[i] + 960 - 420) / (1500 - 420),
                                                    (1080 * rh_points[i + 1] + 540) / 1080,
                                                    (rh_points[2] - rh_points[i + 2]) / 10]
                                                   for i in range(4, len(rh_points), 4)], dtype=np.float32)

                        preFrameCoordP = preFrameCoordP[[i for i in range(19) if (0 <= i <= 7) or (17 <= i <= 18)]]

                        frameCoordL=preFrameCoordL.reshape(5,4,3).transpose(1,0,2)[::-1]
                        frameCoordR=preFrameCoordR.reshape(5,4,3).transpose(1,0,2)[::-1]
                        frameCoordP=preFrameCoordP.T
                        wordCoordL = np.append(wordCoordL, [frameCoordL], axis=0)
                        wordCoordR = np.append(wordCoordR, [frameCoordR], axis=0)
                        wordCoordP = np.append(wordCoordP, [frameCoordP], axis=0)
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_path}: {e}")
                    break
            label=[word_num,int(person),name]
            word_output_path = os.path.join(person_output_path, f'{label[0]}.npz')
            np.savez(word_output_path, wordCoordL=wordCoordL, wordCoordR=wordCoordR, wordCoordP=wordCoordP, label=label)
            print(f"Saved {word_output_path}")
        else:
            continue

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


