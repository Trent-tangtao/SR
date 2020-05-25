import numpy as np
from scipy import misc
from os import listdir
from os.path import join
import cv2
import math


# 最后图片数量为218848
def process_data():
    scale = 3
    input_size = 33
    label_size = 21
    stride = 14
    padding_size = int((input_size - label_size) / 2)
    count = 0
    data_file = './data/Train'
    train_file = "./train_data"
    for f in listdir(data_file):
        f = join(data_file, f)
        origin = misc.imread(f, mode='YCbCr')
        # 也可以用opencv读转化一下
        w, h, c = origin.shape
        # 将图片改成scale的倍数，方便缩放
        w -= w % scale
        h -= h % scale
        image = origin[0:w, 0:h, :]
        # 用双三次插值先缩再放大
        scaled = misc.imresize(image, 1.0 / scale, 'bicubic')
        scaled = misc.imresize(scaled, scale / 1.0, 'bicubic')
        # 裁取33x33的训练，21x21的label
        for i in range(0, w - input_size + 1, stride):
            for j in range(0, h - input_size + 1, stride):
                sub_img = scaled[i:i + input_size, j:j + input_size, :]
                sub_img_label = image[i + padding_size:i + label_size + padding_size,
                                j + padding_size:j + label_size + padding_size, :]
                misc.imsave(join(train_file, 'train', str(count) + ".bmp"), sub_img)
                misc.imsave(join(train_file, 'label', str(count) + ".bmp"), sub_img_label)
                count += 1

    print("done!")


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return rgb.dot(xform.T)