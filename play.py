from keras.models import model_from_json
import numpy as np
import cv2
import random

def prepImg(pth):
    return cv2.resize(pth,(300,300)).reshape(1,300,300,3)

with open('model.json', 'r') as f:
    loaded_model_json = f.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


shape_to_label = {'A':np.array([1.,0.,0.]),'B':np.array([0.,1.,0.]),'C':np.array([0.,0.,1.])}
arr_to_shape = {np.argmax(shape_to_label[x]):x for x in shape_to_label.keys()}

cap = cv2.VideoCapture(0)

NUM_ROUNDS = 30
bplay = ""


while True:
    ret , frame = cap.read()
    frame = frame = cv2.putText(frame,"Press Space to start",(160,200),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
    cv2.imshow('Who is it?',frame)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break

while True:
    pred = ""
    ret,frame = cap.read()
    
    predict = loaded_model.predict(prepImg(frame[50:350, 150:450]))
    print(predict, np.argmax(predict))
    pred = arr_to_shape[np.argmax(predict)]

    cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
    frame = cv2.putText(frame,pred,(150,140),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
    frame = cv2.putText(frame,"Press q to quit",(190,200),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
    cv2.imshow('Who is it?',frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
