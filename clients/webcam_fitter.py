import sys
import site
import numpy as np
from os import path

dirname = path.dirname(path.abspath(__file__))
site.addsitedir(path.join(dirname, '..'))
from facefit.ert.tree import RegressionTree

import cv2
import menpo
import hickle
import menpodetect

def add_landmarks(mat, shape):
    for i in xrange(0, 68):
        cv2.circle(mat, center=(int(shape.points[i][1]), int(shape.points[i][0])), radius=3, color=(0,255,0), thickness=-1)

model = hickle.load(sys.argv[1], safe=False)
face_detector = menpodetect.dlib.load_dlib_frontal_face_detector()

WIDTH=640
HEIGHT=480

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('/Users/rishabhbattulwar/Desktop/tmp/tracking_arcsoft/kiran_out/preview.mov')
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Num of perturbations of the initial shape within a bounding box.
n_inits = 1

ret, orig = cap.read()
orig = cv2.resize(orig, (HEIGHT, WIDTH)) # needed if you're reading a video
orig_menpo = menpo.image.Image(orig.mean(axis=2)/255.0)

# while True:
while ret:
    _, orig = cap.read()
    orig = cv2.resize(orig, (HEIGHT, WIDTH)) # # needed if you're reading a video
    orig_menpo.pixels[:] = (orig.mean(axis=2)/255.0).reshape(1,WIDTH, HEIGHT)
    bbox = face_detector(orig_menpo)
    _, shapes = model.apply(orig_menpo,(bbox, int(n_inits), None))

    for shape in shapes:
        add_landmarks(orig, shape)
        # Add bounding box around the face.
        for box in bbox:
           a, b = box.bounds()
           cv2.rectangle(orig, (int(a[1]),int(a[0])), (int(b[1]),int(b[0])), (0, 255, 0), 2)

    cv2.imshow('frame', orig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
