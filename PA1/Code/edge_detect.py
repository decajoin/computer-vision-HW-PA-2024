import numpy as np
from PIL import Image
import math

# 使用给定的滤波算子对图像进行卷积运算
# img 为黑白灰度图像数据，h 为图像滤波算子矩阵
def my_image_filter(img, h):
    """
    对图像进行卷积运算

    Args:
        img: 输入的黑白灰度图像数据
        h: 卷积操作所使用的滤波算子矩阵

    Returns:
        res_img: 卷积后的图像数据
    """
    h_rows, h_cols = h.shape  # 获取滤波算子的行数和列数
    padded_img = pad_img(img, h.shape)  # 调用 pad_img 对图像进行边缘填充
    res_img = np.zeros(img.shape)  # 初始化与输入图像大小相同的结果图像

    # 对每个像素点进行卷积计算
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 提取当前像素点周围的区域（大小与滤波算子相同）
            region = padded_img[i:i + h_rows, j:j + h_cols]
            # 计算区域与滤波算子的卷积值，并存储在结果图像中
            res_img[i, j] = np.sum(region * h)

    return res_img  # 返回经过卷积后的图像数据

# 填充图片，保证输出图像和输入图像尺寸保持一致
def pad_img(img, h_shape):
    """
    填充图像以保证输出图像与输入图像尺寸一致

    Args:
        img: 输入图像数据
        h_shape: 滤波算子的形状，用于计算填充的大小

    Returns:
        padded_img: 填充后的图像数据
    """
    pad_height, pad_width = h_shape[0] // 2, h_shape[1] // 2  # 计算填充的行列数
    # 使用 'edge' 模式填充图像边缘，这样边缘值与相邻像素相同
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
    return padded_img  # 返回填充后的图像

# 高斯滤波算子的获取
def gauss2d_smooth_filter(shape=(3, 3), sigma=0.5):
    """
    获取高斯滤波算子

    Args:
        shape: 滤波算子的大小，默认为 (3, 3)
        sigma: 高斯函数的标准差，默认为 0.5

    Returns:
        h: 生成的高斯滤波算子矩阵
    """
    # 确定滤波算子的半径
    m, n = [(ss - 1.) / 2. for ss in shape]
    # 创建 Y 和 X 方向的坐标网格
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    # 计算高斯函数值
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    # 设置小于一个最小值的滤波值为零，以提高精度
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    # 归一化高斯核，使其总和为1
    if sumh != 0:
        h /= sumh

    return h  # 返回生成的高斯滤波算子

# Sobel 算子的获取，用于边缘检测
def sobel_filter():
    """
    获取 Sobel 算子，用于计算图像的梯度

    Returns:
        h_x: Sobel X 方向的滤波算子
        h_y: Sobel Y 方向的滤波算子
    """
    # 定义 Sobel X 和 Y 方向的滤波算子
    h_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # X方向
    h_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) # Y方向
    return h_x, h_y  # 返回 X 和 Y 方向的 Sobel 算子

# 非极大值抑制，细化边缘
def non_max_suppression(img, theta):
    """
    对图像进行非极大值抑制，细化边缘

    Args:
        img: 输入的边缘检测结果图像
        theta: 对应的梯度方向图像

    Returns:
        res: 经过非极大值抑制处理后的图像
    """
    row, col = img.shape  # 获取图像的行数和列数
    res = np.zeros((row, col), dtype=np.int32)  # 初始化结果图像

    # 对每个像素点进行非极大值抑制
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            # 获取当前像素点的梯度方向
            angle = theta[i, j] * 180. / np.pi
            angle = (angle + 180) % 180

            # 判断方向并选择相邻像素作为比较对象
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = (img[i, j + 1], img[i, j - 1])
            elif 22.5 <= angle < 67.5:
                neighbors = (img[i + 1, j - 1], img[i - 1, j + 1])
            elif 67.5 <= angle < 112.5:
                neighbors = (img[i + 1, j], img[i - 1, j])
            elif 112.5 <= angle < 157.5:
                neighbors = (img[i - 1, j - 1], img[i + 1, j + 1])

            # 仅保留比相邻像素更大的值，其他置零
            if img[i, j] >= neighbors[0] and img[i, j] >= neighbors[1]:
                res[i, j] = img[i, j]

    return res  # 返回经过非极大值抑制处理后的图像

if __name__ == '__main__':
    # 打开图像并转换为灰度图像
    img = Image.open('./Data/img01.jpg').convert('L')
    img_data = np.asarray(img, dtype="int32")  # 将图像转换为数组形式

    # 使用高斯滤波算子对图像进行平滑处理
    gauss_filter = gauss2d_smooth_filter(shape=(10, 10), sigma=3.0)
    smoothed_img = my_image_filter(img_data, gauss_filter)

    # 使用 Sobel 算子进行边缘检测，分别计算 X 和 Y 方向的梯度
    sobel_x, sobel_y = sobel_filter()
    grad_x = my_image_filter(smoothed_img, sobel_x)  # X方向梯度
    grad_y = my_image_filter(smoothed_img, sobel_y)  # Y方向梯度

    # 对 x 和 y 轴的梯度进行整合
    gradient_magnitude = np.hypot(grad_x, grad_y)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255  # 归一化到 0-255
    theta = np.arctan2(grad_y, grad_x)  # 计算梯度方向

    # 使用非极大值抑制处理梯度幅值图像
    nms_img = non_max_suppression(gradient_magnitude, theta)
    # 将非极大值抑制后的图像转换为PIL格式并保存
    nms_img = Image.fromarray(nms_img).convert("L")
    nms_img.save('./Result/img_edge_thin.jpg')
