import cv2
import math
import numpy as np
import tensorflow as tf


def psnr(target, ref):
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.0))
    return 20 * math.log10(255. / rmse)


def psnr1(tf_img1, tf_img2):
    return tf.image.psnr(tf_img1, tf_img2, max_val=255)


def test_psnr():
    im1 = cv2.imread("sample/test_origin.bmp")
    im2 = cv2.imread("sample/test_input.bmp")
    im3 = cv2.imread("sample/test_res.bmp")
    print("bicubic:" + str(psnr(im1, im2)) + "; SRCNN:" + str(psnr(im1, im3)))

test_psnr()




# loss: 134.4972 - acc: 0.0980
# bicubic:22.95107160851228; SRCNN:22.245613823966078
#  loss: 131.0631 - acc: 0.0998
# bicubic:22.95107160851228; SRCNN:22.69830166368701
# loss 12? acc  0.1?
# bicubic:22.95107160851228; SRCNN:23.280767702686013
# loss: 126.4816 - acc: 0.1020
# bicubic:22.95107160851228; SRCNN:23.35335428334581
# loss: 124.3100 - acc: 0.1030
# bicubic:22.95107160851228; SRCNN:23.468930070987497

