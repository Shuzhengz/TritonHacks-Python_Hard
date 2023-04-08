import cv2

from image_processing import Image_Processing
from matplotlib import pyplot as plt

img = cv2.imread("image.jpg")

stop_data = cv2.CascadeClassifier('stop_data.xml')

test = Image_Processing(img, stop_data)

image = test.get_img_rgb()

if len(test.get_data()) != 0:
    for (x, y, width, height) in test.get_data():
        cv2.rectangle(image, (x, y), (x + height, y + width), (0, 255, 0), 5)

plt.subplot(1, 1, 1)
plt.imshow(image)
plt.show()