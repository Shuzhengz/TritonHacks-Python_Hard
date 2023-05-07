# Method 1: using the Cascade Classifier to detect patterns using haar cascade
import cv2

from image_processing import Image_Processing
from matplotlib import pyplot

img = cv2.imread("image2.jpg")

stop_data = cv2.CascadeClassifier('face_data.xml')

# manually adjust depending on picture size
minSize = 80

# generate own haar cascade
# https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/

test = Image_Processing(img, stop_data)

image = test.get_img_rgb()

rects = test.get_data(minSize)

if len(rects) != 0:
    for (x, y, width, height) in rects:
        cv2.rectangle(image, (x, y), (x + height, y + width), (0, 255, 0), 5)

pyplot.imshow(image)
pyplot.show()
