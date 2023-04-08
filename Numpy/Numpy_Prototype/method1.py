import numpy as np
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt

d_im = cv2.imread('image1.jpg').astype("float64")

normals = np.array(d_im, dtype="float32")
h,w,d = d_im.shape

print("Processing")

for i in range(1,w-1):
  for j in range(1,h-1):
    t = np.array([i,j-1,d_im[j-1,i,0]],dtype="float64")
    f = np.array([i-1,j,d_im[j,i-1,0]],dtype="float64")
    c = np.array([i,j,d_im[j,i,0]] , dtype = "float64")
    d = np.cross(f-c,t-c)
    n = d / np.sqrt((np.sum(d**2)))
    normals[j,i,:] = n

cv2.imwrite("normal.jpg",normals*255)

print("done")
