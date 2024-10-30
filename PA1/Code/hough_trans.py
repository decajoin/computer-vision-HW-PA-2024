import numpy as np
import scipy.ndimage as ndi
import math
from PIL import Image


def hough_line(img, value_threshold=5):
    # Your implemention


def hough_line_peaks(hspace, thetas, rhos, threshold=None):
    # Your implemention


if __name__ == '__main__':
    imgpath = 'img_edge_thin.jpg'
    img = Image.open(imgpath)
    img_data = np.asarray(img, dtype="int32")
    
    # 对图像边缘检测的结果做霍夫变换
    # Your implemention

    # 你可以使用自己熟悉的绘图库
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(img_data, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ori_img = Image.open('../data/img01.jpg')
    ax[1].imshow(ori_img, cmap=cm.gray)
    ax[1].set_ylim((img_data.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    # 在ax[1]中绘制找到的直线
    # Your implemention

    plt.tight_layout()
    plt.show()
