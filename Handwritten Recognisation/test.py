import numpy as np
import cv2


cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)

    img = img / 255
    return img


while True:
    success,imgOriginal=cap.read()
    img=np.asarray(imgOriginal)
    img=cv2.resize(img,(255,255))
    img=preProcessing(img)
    cv2.imshow("hi",img)
    img=img.reshape(180,180,3)
    classIndex=int(model.predict(img))
    print(classIndex)

    if cv2.waitKey(1) &0xFF==ord('q'):
        break




