import numpy as np
import cv2
import os

"""以下是准备数据的部分"""
"""把我们三个人的数据整理好，并加上标签，这样在训练的时候，机器才知道照片里的是谁"""

DATA_PATH = "."

## 我们三个人的代号
people = ['A', 'B', 'C'] 
## 把我们三个人分成三个不同的类型 
# A - [1.,0.,0.] # 1. --> 1.0 浮点数
# B - [0.,1.,0.]
# C - [0.,0.,1.]
shape_to_label = {'A':np.array([1.,0.,0.]),'B':np.array([0.,1.,0.]),'C':np.array([0.,0.,1.])}
arr_to_shape = {np.argmax(shape_to_label[x]):x for x in shape_to_label.keys()}


# 我们的数据一共有两部分组成
## imgData这个是用来装着图片信息的
imgData = list()
# validationData = list()
## label这个是用来装着是谁的（标记）
labels = list()

for dr in os.listdir(DATA_PATH): # 拿到当前目录下的全部文件
    if dr not in people: # 只处理我们名字的文件夹
        continue # 不是我们名字文件夹的照片不处理
    label = shape_to_label[dr] #
    i = 0

    # 这里拿到的是我们各自照片目录面的全部照片
    pictures = os.listdir(os.path.join(DATA_PATH, dr))
    for pic in pictures:
        if not pic.endswith("jpg"):
            continue
        path = os.path.join(DATA_PATH,dr+'/'+pic)

        # 把照片读进来，用img变量保存
        img = cv2.imread(path)

        # if i > len(pictures) * 4 / 5:
        #     validationData.append([img, label])
        #     validationData.append([cv2.flip(img, 1), label])  # horizontally flipped image
        #     validationData.append([cv2.resize(img[50:250, 50:250], (300, 300)), label])  # zoom : crop in and resize
        # else:
        imgData.append([img, label])
        imgData.append([cv2.flip(img, 1), label])  # horizontally flipped image
        imgData.append([cv2.resize(img[50:250, 50:250], (300, 300)), label])  # zoom : crop in and resize
        i += 3


# 我们要打乱一下照片的顺序，不同的学习顺序，有时候会对模型产生影响
np.random.shuffle(imgData)

imgData,labels = zip(*imgData)
imgData = np.array(imgData)
labels = np.array(labels)

# validationData, validationLabel = zip(*validationData)
# validationData = np.array(validationData)
# validationLabel = np.array(validationLabel)

"""这里开始是我们定义模型"""
"""这里选用了一个基础模型DenseNet121"""

""" 
这个模型的首作和二作都是华人噢
康奈尔大学博士后黄高博士Gao Huang、
清华大学本科生刘壮Zhuang Liu、
Facebook 人工智能研究院研究科学家 Laurens van der Maaten 
康奈尔大学计算机系教授 Kilian Q. Weinberger
"""

from keras.models import Sequential,load_model
from keras.layers import Dense,MaxPool2D,Dropout,Flatten,Conv2D,GlobalAveragePooling2D,Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers.legacy import Adam
from keras.applications.densenet import DenseNet121

# imgData = tf.keras.applications.densenet.preprocess_input(imgData)


# 定义这个模型最重要的就是，class=3，就是输入的数据一共分成三类（思考题：那三类呢？）
# input_shape 说明了我们输入的数据的格式 (300,300,3), 前两个300表示照片的尺寸，最后一个3表示三个颜色通道（R、G、B）
densenet = DenseNet121(include_top=False, weights='imagenet', classes=3,input_shape=(300,300,3))
densenet.trainable=True

# 好啦， 我们在DenseNet121前后都加了几层
# 比较重要的内容
# 1. Dense 3，我们最终是要模型告诉我们照片是三个人里面的谁，所以我们期望模型看到照片以后激活三个中的一个
def genericModel(base):
    model = Sequential()
    model.add(base)
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['acc'])
    # model.summary()
    return model

dnet = genericModel(densenet)


# 这里是告诉机器我们期望怎么学习
# 我们期望他学好了就记录到model.h5里面
# 他通过 val_acc 来监督自己的学习，判断自己这次学习的结果好还是不好
# save_best_only=True，只保存最好的学习效果
checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True,
    mode='auto'
)

# EarlyStopping决定他提前放弃自我
# patience = 3 意思就是三次学习都没有突破的话，就放弃
es = EarlyStopping(patience = 3)

# dnet.fit() 这个函数就是叫模型开始学习
# x 就是图片信息，y就是我们做好的标记 labels
# batch_size 就是每次看多少张图片
# epoch 就是全部的图片看多少次 
# validation_split 模型预留多少数据来做自我检查
history = dnet.fit(
    x=imgData,
    y=labels,
    batch_size = 10,
    epochs=8,
    callbacks=[checkpoint,es],
    # validation_data=(validationData, validationLabel)
    validation_split=0.2
)

# dnet.save_weights('model.h5')

with open("model.json", "w") as json_file:
    json_file.write(dnet.to_json())
