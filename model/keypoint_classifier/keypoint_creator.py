import csv
import copy
import cv2
import mediapipe as mp
import numpy as np
import itertools
import os
from tensorflow import keras

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.5)

def process_frame(image, label, class_names):
  image = np.uint8(image)

  # Flip image around y-axis for correct handedness output (see above).
  image = cv2.flip(image, 1)

  # Convert the BGR image to RGB before processing.
  results = hands.process(image)

  # Print handedness and draw hand landmarks on the image.
  # print('Handedness:', results.multi_handedness)

  if not results.multi_hand_landmarks or not results.multi_hand_world_landmarks:
    return []

  # image_height, image_width, _ = image.shape
  # annotated_image = image.copy()
  # for hand_landmarks in results.multi_hand_landmarks:
  #   mp_drawing.draw_landmarks(
  #       annotated_image,
  #       hand_landmarks,
  #       mp_hands.HAND_CONNECTIONS,
  #       mp_drawing_styles.get_default_hand_landmarks_style(),
  #       mp_drawing_styles.get_default_hand_connections_style())
  # cv2.imwrite('./tmp/annotated_image' + str(label) + '.png', cv2.flip(annotated_image, 1))

  debug_image = copy.deepcopy(image)
  for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
    pre_processed_landmark_list = pre_process_landmark(landmark_list)
    return [class_names[label], *pre_processed_landmark_list]

def main():
  make_backup()
  csv_path = 'keypoint.csv'
  dataset = keras.utils.image_dataset_from_directory('..\\..\\..\\training_data\\signs\\Alphabet\\Training Set', labels='inferred', batch_size=1, color_mode='rgb')
  class_names = dataset.class_names

  file = open(csv_path, "w")
  file.close()

  output = []

  count = 1
  for images, labels in dataset.take(-1):
    if count % 500 == 0:
      print(f"{count} / {len(dataset)}")
    count += 1
    image = images[0].numpy()
    label = labels[0]

    results = process_frame(image, label, class_names)
    if len(results) != 0:
      output.append(results)

  with open(csv_path, 'a', newline="") as f:
    writer = csv.writer(f)

    for row in output:
      # writer.writerow([class_names[label], *pre_processed_landmark_list])
      writer.writerow(row)
    f.close()

def make_backup():
  if os.path.exists("keypoint.csv"):
    os.rename("keypoint.csv", "keypoint_backup/keypoint.csv")

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def calc_landmark_list(image, landmarks):
  image_width, image_height = image.shape[1], image.shape[0]

  landmark_point = []

  # Keypoint
  for _, landmark in enumerate(landmarks.landmark):
      landmark_x = min(int(landmark.x * image_width), image_width - 1)
      landmark_y = min(int(landmark.y * image_height), image_height - 1)

      landmark_point.append([landmark_x, landmark_y])

  return landmark_point

if __name__ == "__main__":
  main()
