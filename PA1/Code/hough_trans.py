import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import math

def hough_line(img, value_threshold=5):
    """
    对图像进行霍夫变换来检测直线

    Args:
        img: 输入图像(边缘检测后的结果)
        value_threshold: 边缘像素点的阈值

    Returns:
        hspace: 霍夫空间的累加器数组
        thetas: 角度数组(弧度制)
        rhos: 距离数组
    """
    # 获取图像尺寸
    height, width = img.shape

    # 计算图像对角线长度作为rho的最大值
    max_distance = int(np.ceil(np.sqrt(height**2 + width**2)))

    # 初始化角度数组(-90到90度,转换为弧度)
    thetas = np.deg2rad(np.arange(-90, 90, 1))

    # 初始化rho数组(-max_distance到max_distance)
    rhos = np.arange(-max_distance, max_distance)

    # 初始化投票空间(累加器数组)
    hspace = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # 获取边缘点的坐标
    y_idxs, x_idxs = np.nonzero(img > value_threshold)

    # 对每个边缘点进行投票
    for i in range(len(y_idxs)):
        y = y_idxs[i]
        x = x_idxs[i]

        # 对每个角度计算对应的rho值
        for theta_idx in range(len(thetas)):
            theta = thetas[theta_idx]
            # 使用霍夫变换的标准方程: rho = x*cos(theta) + y*sin(theta)
            rho = x * np.cos(theta) + y * np.sin(theta)
            # 找到最接近的rho值的索引
            rho_idx = np.argmin(np.abs(rhos - rho))
            # 在累加器数组中投票
            hspace[rho_idx, theta_idx] += 1

    return hspace, thetas, rhos

def hough_line_peaks(hspace, thetas, rhos, threshold=None):
    """
    从霍夫变换结果中找出峰值，代表检测到的直线

    Args:
        hspace: 霍夫变换的累加器数组
        thetas: 角度数组
        rhos: 距离数组
        threshold: 最低投票数阈值

    Returns:
        peak_thetas: 峰值对应的角度
        peak_rhos: 峰值对应的距离
    """
    # 如果没有设置阈值，使用投票数最大值的50%作为阈值
    if threshold is None:
        threshold = np.max(hspace) * 0.5

    # 创建一个掩码数组来标记已处理的区域
    processed_mask = np.zeros_like(hspace, dtype=bool)

    peak_thetas = []
    peak_rhos = []

    # 设置NMS窗口大小
    window_size = 15

    while True:
        # 在未处理区域中找到最大值
        if np.max(hspace[~processed_mask]) < threshold:
            break

        # 找到当前最大值的位置
        max_idx = np.argmax(hspace[~processed_mask])
        max_position = np.where(~processed_mask)[0][max_idx], np.where(~processed_mask)[1][max_idx]

        # 获取周围区域
        y_start = max(0, max_position[0] - window_size//2)
        y_end = min(hspace.shape[0], max_position[0] + window_size//2 + 1)
        x_start = max(0, max_position[1] - window_size//2)
        x_end = min(hspace.shape[1], max_position[1] + window_size//2 + 1)

        # 如果当前点确实是局部最大值
        window = hspace[y_start:y_end, x_start:x_end]
        if hspace[max_position] == np.max(window):
            peak_thetas.append(thetas[max_position[1]])
            peak_rhos.append(rhos[max_position[0]])

            # 标记该区域为已处理
            processed_mask[y_start:y_end, x_start:x_end] = True
        else:
            processed_mask[max_position] = True

    return np.array(peak_thetas), np.array(peak_rhos)

if __name__ == '__main__':
    # 读取边缘检测后的图像
    imgpath = './Result/img_edge_thin.jpg'
    img = Image.open(imgpath)
    img_data = np.asarray(img, dtype="int32")

    # 对图像进行霍夫变换
    hspace, thetas, rhos = hough_line(img_data, value_threshold=5)

    # 检测峰值点
    peak_thetas, peak_rhos = hough_line_peaks(hspace, thetas, rhos)

    # 绘图展示结果
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(img_data, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ori_img = Image.open('./Data/img01.jpg')
    ax[1].imshow(ori_img, cmap=cm.gray)
    ax[1].set_ylim((img_data.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    # 在原始图像上绘制检测到的直线
    for theta, rho in zip(peak_thetas, peak_rhos):
        # 计算直线的端点
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 800 * (-b))
        y1 = int(y0 + 800 * (a))
        x2 = int(x0 - 800 * (-b))
        y2 = int(y0 - 800 * (a))

        # 绘制直线
        ax[1].plot([x1, x2], [y1, y2], 'r-')

    # 保存绘制的结果图
    plt.tight_layout()

    # 指定保存的文件路径和文件名
    plt.savefig('./Result/detected_lines.png')

    plt.tight_layout()
    plt.show()