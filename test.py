from keras.models import model_from_json
import numpy as np
from skimage import io
import cv2
import random
import os


def prepImg(pth):
    return cv2.resize(pth, (300, 300)).reshape(1, 300, 300, 3)


def updateScore(play, bplay, p, b):
    winRule = {'rock': 'scissor', 'scissor': 'paper', 'paper': 'rock'}
    if play == bplay:
        return p, b
    elif bplay == winRule[play]:
        return p + 1, b
    else:
        return p, b + 1


with open('model.json', 'r') as f:
    loaded_model_json = f.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
LABEL = ['rock','paper','scissor']

for dr in os.listdir("."):
    if dr in ['rock','paper','scissor']:
        for pic in os.listdir(os.path.join(".", dr)):
            if not pic.endswith("jpg"):
                continue
            path = os.path.join(".",dr + "/" + pic)

            img = cv2.imread(path)

            ret = loaded_model.predict(prepImg(img))
            if LABEL[np.argmax(ret)] != dr:
                print("predict:", LABEL[np.argmax(ret)], "expected:", dr, ret)
            # print(ret, np.argmax(ret))
            #
            #
            # print(path, ret)