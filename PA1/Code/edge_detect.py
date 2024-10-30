import numpy as np
from PIL import Image
import math


# 入参为numpy二维数组
def my_image_filter(img, h):
    res_img = np.zeros(img.shape)

    # Your implemention

    return res_img


def pad_img(img, h_shape):
    # Your implemention


def gauss2d_smooth_filter(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def smooth_img_with_gauss_filter(img, sigma):
    # Your implemention


# theta对应课件中的direction
def non_max_suppression(img, theta):
    row, col = img.shape
    res = np.zeros((row, col), dtype=np.int32)
    
    # Your implemention

    return res


if __name__ == '__main__':
    # convert('L')用于将RGB转为黑白
    img = Image.open('../data/img01.jpg').convert('L')
    img_data = np.asarray(img, dtype="int32")

    # 使用高斯滤波算子对图像进行平滑处理
    # Your implemention
    
    # 使用Sobel算子进行边缘检测
    # Your implemention
    
    # 使用NMS处理，并保留图片，设置文件名为img_edge_thin.jpg
    # Your implemention
