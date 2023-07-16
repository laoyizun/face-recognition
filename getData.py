import numpy as np
import cv2
import os
import sys

PATH = os.getcwd()+"/"
cap = cv2.VideoCapture(1)

label = sys.argv[1]

SAVE_PATH = os.path.join(PATH, label)

try:
    os.mkdir(SAVE_PATH)
except FileExistsError:
    pass

# 这里决定了我们要为每个人保存多少照片给机器学习
maxCount = 30


print("Hit Space to Capture Image")

count = 0
# 每拍一张count就+1，直到count == maxCount
while count < maxCount:
    ret, frame = cap.read()
    cv2.imshow('Get Data : '+label,frame[50:350,150:450])

    ## 按了空格键
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite(SAVE_PATH+'/'+label+'{}.jpg'.format(count),frame[50:350,150:450])
        print(SAVE_PATH+'/'+label+'{}.jpg Captured'.format(count))
        ## 这样就拍好一张了
        count += 1

cap.release()
cv2.destroyAllWindows()
