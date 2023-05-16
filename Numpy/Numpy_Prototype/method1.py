import numpy as np
import mediapipe as mp
from scipy import ndimage
import cv2
from matplotlib import pyplot as plt

# Background Removal
img = cv2.imread('image1.jpg')
background = cv2.imread('bg.jpg')

height, width, channel = img.shape

mpSelfieSegmentation = mp.solutions.selfie_segmentation
segmentation = mpSelfieSegmentation.SelfieSegmentation(model_selection=1)

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = segmentation.process(imgRGB)
# extract segmented mask
mask = results.segmentation_mask

# it returns true or false where the condition applies in the mask
condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 1e-5

# resize the background image to the same size of the original frame
background = cv2.resize(background, (width, height))

output_image = np.where(condition, img, background)

# show outputs
plt.imshow(output_image)
plt.show()


# Normal Map Generation
d_im = output_image.astype("float64")

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
