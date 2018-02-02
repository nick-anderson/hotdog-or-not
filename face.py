import cv2
import os

os.system('curl %s -o image'%'http://skorthodontics.com/images/family-home.png')
# face_cascade = cv2.CascadeClassifier('haarcascades/haarcascades.xml')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
img = cv2.imread('image',cv2.IMREAD_GRAYSCALE)
faces = face_cascade.detectMultiScale(img, 1.05, 4,minSize=(200,200))
# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
img = cv2.imread('image')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()