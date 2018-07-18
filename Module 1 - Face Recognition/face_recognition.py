# imports
import cv2
from skimage.io import imshow

# load the cascades
face_cascade = cv2.CascadeClassifier("./Module 1 - Face Recognition/cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./Module 1 - Face Recognition/cascades/haarcascade_eye.xml")

# creating the detector
def detect(img_gray, img):
    faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)

    # iterate through each of the detected faces and look for eyes
    for (x, y, w, h) in faces:
        # draw a rectangle around the face
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)

        # crop the original image to the part where the face was detected
        roi_gray = img_gray[y: y+h, x: x+w]
        roi_img = img[y: y+h, x: x+w]

        # now detect the eyes
        eyes = eye_cascade.detectMultiScale(image=roi_gray, scaleFactor=1.1, minNeighbors=3)

        # now loop through the eyes and draw rectangles
        for (xe, ye, we, he) in eyes:
            cv2.rectangle(img=roi_img, pt1=(xe, ye), pt2=(xe+we, ye+he), color=(0, 255, 0), thickness=2)

    return img

img = cv2.imread("./Module 1 - Face Recognition/images/img_1.png")
img_grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

out = detect(img_grey, img)
imshow(out[:, :, ::-1])
