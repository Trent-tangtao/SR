import cv2
import matplotlib.pyplot as plt
from scipy import misc
from utils import ycbcr2rgb
import numpy as np

# im1 = misc.imread("test.bmp",)
# plt.imshow(im1)
# plt.show()

# im1 = misc.imread("test.bmp", mode='YCbCr')
# print(im1.dtype)
# plt.imshow(im1)
# plt.show()

# im1 = ycbcr2rgb(im1)
# plt.imshow(im1)
# plt.show()

im1 = misc.imread("train_data/label/12396.bmp",)
print(im1.shape)
im1 = ycbcr2rgb(im1)
im1 = np.array(im1,dtype='uint8')
plt.imshow(im1)
plt.show()