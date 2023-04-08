import cv2


class Image_Processing:
    def __init__(self, img, data):
        self.img = img
        self.data = data

    def get_img_grey(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def get_img_rgb(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def get_data(self):
        return self.data.detectMultiScale(self.get_img_grey(), minSize=(20, 20))
