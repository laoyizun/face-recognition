from keras.models import model_from_json
import numpy as np
import cv2
import os


def prepImg(pth):
    return cv2.resize(pth, (300, 300)).reshape(1, 300, 300, 3)

with open('model.json', 'r') as f:
    loaded_model_json = f.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
LABEL = ['A','B','C']

for dr in os.listdir("."):
    if dr in LABEL:
        for pic in os.listdir(os.path.join(".", dr)):
            if not pic.endswith("jpg"):
                continue
            path = os.path.join(".",dr + "/" + pic)

            img = cv2.imread(path)

            # loaded_model.predict() 这个函数就是让模型判断一张图片更像是哪一类
            ret = loaded_model.predict(prepImg(img))


            if LABEL[np.argmax(ret)] != dr:
                print("predict:", LABEL[np.argmax(ret)], "expected:", dr, ret)