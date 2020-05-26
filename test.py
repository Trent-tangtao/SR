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
    # return (np.abs(target - ref) ** 2).mean()


def psnr1(tf_img1, tf_img2):
    return tf.image.psnr(tf_img1, tf_img2, max_val=255)


def test_psnr():
    im1 = cv2.imread("sample/test_origin.bmp")
    im2 = cv2.imread("sample/test_input.bmp")
    im3 = cv2.imread("sample/test_res.bmp")
    print("bicubic:" + str((psnr(im1, im2))) +  "; SRCNN:" + str((psnr(im1, im3))))
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print("bicubic:" + str(sess.run(psnr1(im1, im2))) +
    #           "; SRCNN:" + str(sess.run(psnr1(im1, im3))))

if __name__ == '__main__':
    test_psnr()



