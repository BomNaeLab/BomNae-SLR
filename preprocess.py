import os
import json
import numpy as np

# train_dir = 'D:/signData'
train_dir = '/home/root/data/signData'
output_dir = f"{train_dir}/nptxt"
json_folder_path = f'{train_dir}/train/label/landmark'


def transShape(array, x, y, flag=True):
    """Reshape and reformat the coordinates."""
    coords = [array[:, i].reshape(x, y) for i in range(3)]  # List of reshaped x, y, z arrays
    if flag:
        return np.stack([c.T[::-1] for c in coords])  # Transform if flag is True
    else:
        return np.stack(coords, axis=0)  # No transformation, stack directly


for person in os.listdir(json_folder_path):
    person_output_path = os.path.join(output_dir, person)
    os.makedirs(person_output_path, exist_ok=True)
    for word in os.listdir(os.path.join(json_folder_path, person)):
        if "F" in word:
            wordCoordL = np.empty((0, 3, 4, 5))  # Initialize arrays for keypoints
            wordCoordR = np.empty((0, 3, 4, 5))
            wordCoordP = np.empty((0, 3, 1, 10))

            for frame in os.listdir(os.path.join(json_folder_path, person, word)):
                file_path = os.path.join(json_folder_path, person, word, frame)

                try:
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        lh_points = data['people']['hand_left_keypoints_3d']
                        rh_points = data['people']['hand_right_keypoints_3d']
                        p_points = data['people']['pose_keypoints_3d']

                        # Create normalized arrays for left, right hands, and pose keypoints
                        preFrameCoordP = np.array([[960 * p_points[j] + 960, 1080 * p_points[j + 1] + 540,
                                                    (p_points[32 + 2] - p_points[j + 2]) / 10]
                                                   for j in range(0, len(p_points), 4)], dtype=np.float32)

                        preFrameCoordL = np.array([[(960 * lh_points[i] + 960 - 420) / (1500 - 420),
                                                    (1080 * lh_points[i + 1] + 540) / 1080,
                                                    (lh_points[2] - lh_points[i + 2]) / 10]
                                                   for i in range(4, len(lh_points), 4)], dtype=np.float32)

                        preFrameCoordR = np.array([[(960 * rh_points[i] + 960 - 420) / (1500 - 420),
                                                    (1080 * rh_points[i + 1] + 540) / 1080,
                                                    (rh_points[2] - rh_points[i + 2]) / 10]
                                                   for i in range(4, len(rh_points), 4)], dtype=np.float32)

                        # Filter specific pose keypoints
                        preFrameCoordP_temp = preFrameCoordP[[i for i in range(21) if (0 <= i <= 7) or (17 <= i <= 18)]]

                        # Transform shapes
                        frameCoordL = transShape(preFrameCoordL, 5, 4)
                        frameCoordR = transShape(preFrameCoordR, 5, 4)
                        frameCoordP = transShape(preFrameCoordP_temp, 1, 10, False)

                        # Append frame data to the word arrays
                        wordCoordL = np.append(wordCoordL, [frameCoordL], axis=0)
                        wordCoordR = np.append(wordCoordR, [frameCoordR], axis=0)
                        wordCoordP = np.append(wordCoordP, [frameCoordP], axis=0)

                        word_output_path = os.path.join(person_output_path, f'{word}.npz')

                        # Save multiple arrays for the word (Left hand, Right hand, Pose)
                        np.savez(word_output_path, wordCoordL=wordCoordL, wordCoordR=wordCoordR, wordCoordP=wordCoordP)
                        print(f"Saved {word_output_path}")
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_path}: {e}")
                    break

            # Final word data is stored in wordCoordL, wordCoordR, and wordCoordP
