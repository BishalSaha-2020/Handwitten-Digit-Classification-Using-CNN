import os

from pip._internal.cli.cmdoptions import verbose

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle



images = []
classNo = []

# Path
path = 'mydata'
mylist = os.listdir(path)

print(len(mylist))
noOfClasses = len(mylist)

# Checking the Data
for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
print()
print(len(images))
print(classNo)

# Checking shapes
images = np.array((images))
classNo = np.array(classNo)
print(images.shape)
# print(classNo.shape)

# Spliting the Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))
print(numOfSamples)


print("Hello")


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img / 255
    return img


# visualization of images

# img=preProcessing(X_train[30])
# img=cv2.resize(img,(300,300))
# cv2.imshow("preprocessed",img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

print((X_train).shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
print((X_train).shape)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

print("bro bro")
def myModel():
    noOffilters = 60
    sizeOffilter = (5, 5)
    sizeOffilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500
    model = Sequential()
    model.add((Conv2D(noOffilters, sizeOffilter, input_shape=(32, 32, 1), activation='relu')))
    model.add((Conv2D(noOffilters, sizeOffilter, activation='relu')))
    model.add(MaxPool2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOffilters // 2, sizeOffilter2, activation='relu')))
    model.add((Conv2D(noOffilters // 2, sizeOffilter2, activation='relu')))
    model.add(MaxPool2D(pool_size=sizeOfPool))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


    return model


model = myModel()
print(model.summary())
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=50),epochs=1,steps_per_epoch=50,validation_data=(X_validation,y_validation),shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training,validation'])
plt.title('loss')
plt.plot()
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training,validation'])
plt.title('accuracy')
plt.xlabel('no of epoch')
plt.show()

score=model.evaluate((X_test,y_test))
print("Test Score",score[0])
print("Test Accuracy",score[1])



cap=cv2.VideoCapture(0)







while True:
    success,imgOriginal=cap.read()
    img=np.asarray(imgOriginal)
    img=cv2.resize(img,(1000,1000))
    img=preProcessing(img)
    cv2.imshow("hi",img)
    img=img.reshapr(1,1000,1000,1)

    classIndex=int(model.predict_classes(img))

    print(classIndex)

    cv2.waitKey(1)






















