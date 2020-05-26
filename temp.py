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

# im1 = misc.imread("train_data/label/12396.bmp",)
# print(im1.shape)
# im1 = ycbcr2rgb(im1)
# im1 = np.array(im1,dtype='uint8')
# plt.imshow(im1)
# plt.show()

im1 = cv2.imread("sample/test_origin.bmp")
im2 = cv2.imread("sample/test_input.bmp")
im3 = cv2.imread("sample/test_res.bmp")
im4 = cv2.imread("sample/test_low.bmp")

w,h,c = im1.shape
w2,h2,c2 =im4.shape
ww=int((w-w2)/2)
hh=int((h-h2)/2)
res_img = np.zeros((w,h*4,3))
res_img[0:w,0:h,0]=im1[:,:,0]
res_img[ww:ww+w2,h+hh:h+hh+h2, 0]=im4[:,:,0]
res_img[0:w,h*2:h*3,0]=im2[:,:,0]
res_img[0:w,h*3:h*4,0]=im3[:,:,0]
res_img[:,:,1]=res_img[:,:,0]
res_img[:,:,2]=res_img[:,:,0]
misc.imsave("sample/res.bmp", res_img)
