# [OpenCV](https://shengyu7697.github.io/python-opencv-resize/)
import os
import cv2
import matplotlib.pyplot as plt

# Use an existing demo image in the repository
file_path = os.path.join("images", "hw4_pic01.PNG")
image = cv2.imread(file_path)
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
print("image size is", image.shape)

cv2.imshow(file_path, image)
cv2.waitKey(0)

# opencv BGR to matplotlib RGB in `imshow`
OPENCV2MATPLOTLIB = False
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if OPENCV2MATPLOTLIB else image

plt.subplots(1, 1)
plt.imshow(img_rgb)
plt.title(file_path)
plt.show()