import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle

# 数据和结果存储路径
data_path = '../Data'
result_dir = "../Result"
category = ['airport', 'auditorium', 'bedroom', 'campus', 'desert', 'football_stadium', 'landscape', 'rainforest']
label_map = {cat: idx for idx, cat in enumerate(category)}  # 类别标签映射
no_clusters = 100  # 设置KMeans聚类的簇数


def get_images_descriptors(detector, image_path_array, ori_labels):
    """
    提取图像的描述子（特征点的描述符）

    :param detector: ORB 或其他特征检测器
    :param image_path_array: 图像路径列表
    :param ori_labels: 对应图像的标签
    :return: 描述子和对应标签的列表
    """
    descriptors = []
    labels = []
    for img_path, label in zip(image_path_array, ori_labels):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
        if image is None:
            continue  # 如果图像无法读取，则跳过
        keypoints, desc = detector.detectAndCompute(image, None)  # 提取特征点及描述符
        if desc is not None:
            descriptors.append(desc)  # 保存描述符
            labels.append(label)  # 保存对应的标签
    return descriptors, labels


def vstack_descriptors(descriptors_list):
    """
    将多个描述符堆叠为一个大的数组

    :param descriptors_list: 各图像的描述符列表
    :return: 垂直堆叠的描述符矩阵
    """
    return np.vstack(descriptors_list)


def cluster_descriptors(descriptors, no_clusters):
    """
    使用KMeans对所有描述子进行聚类，生成视觉词典（视觉码本）

    :param descriptors: 所有图像的描述符
    :param no_clusters: 设置KMeans聚类的簇数
    :return: 聚类模型
    """
    kmeans = KMeans(n_clusters=no_clusters, random_state=42)  # 初始化KMeans
    kmeans.fit(descriptors)  # 对描述符进行聚类
    return kmeans


def extract_features(kmeans, descriptors_list, no_clusters):
    """
    将每个图像的描述子映射到视觉词典，并生成图像的特征向量

    :param kmeans: 聚类模型
    :param descriptors_list: 图像描述符列表
    :param no_clusters: 聚类簇数
    :return: 每张图像的特征向量
    """
    image_count = len(descriptors_list)
    im_features = np.zeros((image_count, no_clusters))  # 初始化特征矩阵
    for idx, desc in enumerate(descriptors_list):
        if desc is not None:
            clusters = kmeans.predict(desc)  # 获取描述子在视觉词典中的簇分布
            for cluster_idx in clusters:
                im_features[idx][cluster_idx] += 1  # 增加该簇的计数
    return im_features


def train_SVC(features, train_labels):
    """
    使用支持向量机（SVM）训练分类器

    :param features: 图像的特征向量
    :param train_labels: 训练集的标签
    :return: 训练好的SVM分类器
    """
    svc = SVC(kernel='rbf', C=1, gamma=0.01, random_state=42)  # 初始化SVM分类器
    svc.fit(features, train_labels)  # 训练分类器
    return svc


if __name__ == '__main__':
    # 预备操作：加载数据和对应的数字标签，并切分训练集和测试集
    image_paths = []  # 存储所有图像的路径
    labels = []  # 存储图像的标签
    for cat in category:
        cat_path = os.path.join(data_path, cat)  # 获取每个类别文件夹路径
        for file_name in os.listdir(cat_path):
            image_paths.append(os.path.join(cat_path, file_name))  # 添加图像路径
            labels.append(label_map[cat])  # 添加对应标签

    # 将数据打乱并分割为训练集和测试集
    combined = list(zip(image_paths, labels))
    np.random.shuffle(combined)
    image_paths, labels = zip(*combined)
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # 构建视觉码本，使用 ORB 算法通过 cv2.ORB_create() 提取特征点及描述子
    orb = cv2.ORB_create()

    # 在测试图像中绘制特征点
    test_image_path = '../Data/desert/sun_acqlitnnratfsrsk.jpg'
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(test_image, None)  # 提取特征点及描述符
    img_with_keypoints = cv2.drawKeypoints(test_image, keypoints, None, color=(0, 255, 0))  # 绘制关键点
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title('Feature Keypoints')

    # 保存绘制了关键点的图片
    keypoints_image_path = os.path.join(result_dir, "keypoints_image.jpg")
    plt.savefig(keypoints_image_path)
    print(f"Keypoints image saved to {keypoints_image_path}")

    plt.show()

    # 批量提取训练集图像的特征描述子
    train_descriptors, train_labels = get_images_descriptors(orb, X_train, y_train)

    # 将描述符堆叠为一个大矩阵
    stacked_descriptors = vstack_descriptors(train_descriptors)

    # 使用KMeans进行聚类，生成视觉词典（视觉码本）
    kmeans = cluster_descriptors(stacked_descriptors, no_clusters)

    # 保存训练好的KMeans模型
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    # 提取训练集的图像特征
    train_features = extract_features(kmeans, train_descriptors, no_clusters)

    # 对特征进行标准化处理
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    # 保存Scaler（标准化处理模型）
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 训练SVM分类器
    svc = train_SVC(train_features, train_labels)

    # 保存训练好的SVM模型
    with open('svc_model.pkl', 'wb') as f:
        pickle.dump(svc, f)

    # 评估分类器：在测试集上进行评估
    test_descriptors, test_labels = get_images_descriptors(orb, X_test, y_test)
    test_features = extract_features(kmeans, test_descriptors, no_clusters)
    test_features = scaler.transform(test_features)  # 对测试特征进行标准化

    # 使用SVM预测测试集的标签
    y_pred = svc.predict(test_features)
    accuracy = accuracy_score(test_labels, y_pred)  # 计算准确率
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # 绘制并保存混淆矩阵
    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(category)), labels=category, rotation=45)
    plt.yticks(ticks=np.arange(len(category)), labels=category)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 在每个格子里显示数值
    thresh = cm.max() / 2.0  # 设置一个阈值，用于判断数值是显示在浅色还是深色背景上
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    confusion_matrix_image_path = os.path.join(result_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_image_path)
    print(f"Confusion matrix saved to {confusion_matrix_image_path}")

    plt.show()
