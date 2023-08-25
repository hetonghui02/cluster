import os
import shutil
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import accuracy_score

# 图片文件夹路径
folder_path = "../picture/picture_spec_contour"
test_folder_path = "../picture/val_picture_contour"
similarity_threshold = 0.9


# 获取所有图片文件名
file_names = [file for file in os.listdir(folder_path) if file.endswith(".jpg")]

# 提取出所有型号
models = list(set([file.split("_")[0] for file in file_names]))

# 聚类数量
num_clusters = len(models)

# 创建"cluster"文件夹
os.makedirs("cluster", exist_ok=True)

# 创建聚类文件夹
cluster_folders = []  # 用于存储聚类文件夹的顺序
for i in range(num_clusters):
    cluster_name = f"{models[i]}"
    cluster_folder = os.path.join("cluster", cluster_name)
    os.makedirs(cluster_folder, exist_ok=True)
    cluster_folders.append((models[i], cluster_folder))  # 将型号和聚类文件夹路径作为元组存储

# 按型号名称排序聚类文件夹列表
cluster_folders.sort(key=lambda x: x[0])

# 复制图片到相应的聚类文件夹并按照规则进行命名
for file_name in file_names:
    model = file_name.split("_")[0]
    cluster_name = f"{model}"
    cluster_folder = [folder for m, folder in cluster_folders if m == model][0]  # 根据型号名称获取聚类文件夹路径
    new_file_name = os.path.join(cluster_folder, file_name)
    shutil.copyfile(os.path.join(folder_path, file_name), new_file_name)

# 加载ResNet50模型进行特征提取


model_name = ResNet50(weights='imagenet', include_top=False, pooling='avg')


# 特征提取函数
def extract_features(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # 保持与ResNet50的输入尺寸一致
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model_name.predict(img)
    return features.flatten()

# cluster文件夹路径
cluster_folder_path = "cluster"  # 根据实际情况进行修改

# 加载聚类文件夹中的图片特征
cluster_features = []
cluster_names = []
for cluster_name in os.listdir(cluster_folder_path):
    if cluster_name != ".DS_Store":  # 排除聚类名为 "DS_Store" 的情况
        cluster_path = os.path.join(cluster_folder_path, cluster_name)

    # 获取聚类文件夹中的图片特征数据
    features_list = []
    for file_name in os.listdir(cluster_path):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(cluster_path, file_name)
            features = extract_features(image_path)  # 使用之前定义的特征提取函数
            features_list.append(features)

    # 计算聚类中心特征值
    cluster_center = np.mean(features_list, axis=0)

    # 将聚类名和聚类中心特征值添加到cluster_names和cluster_features中
    cluster_names.append(cluster_name)
    cluster_features.append(cluster_center)


# 对test文件夹中的新图片进行聚类预测并计算相似度
test_file_names = [file for file in os.listdir(test_folder_path) if file.endswith(".jpg")]

for test_file_name in test_file_names:
    test_image_path = os.path.join(test_folder_path, test_file_name)
    test_features = extract_features(test_image_path)

    # 更新聚类文件夹中的图片特征
    for i, cluster_feature in enumerate(cluster_features):
        cluster_feature = np.vstack((cluster_feature, test_features))
        cluster_features[i] = cluster_feature

    similarity_scores = []
    for cluster_images_features in cluster_features:
        similarity_score = np.mean(cosine_similarity([test_features], cluster_images_features))
        similarity_scores.append(similarity_score)

    max_similarity_indices = np.argsort(similarity_scores)[::-1][:3]  # 获取相似度前三高的聚类索引
    max_similarity_scores = np.array(similarity_scores)[max_similarity_indices]

    print(f"图片 {test_file_name} 的相似度前三高的聚类名和相似度结果：")
    for cluster_index, similarity_score in zip(max_similarity_indices, max_similarity_scores):
        cluster_name = f"{cluster_names[cluster_index]}"
        print(f"聚类索引: {cluster_index}，聚类名称：{cluster_name},相似度: {similarity_score:.4f}")

    if max_similarity_scores[0] > similarity_threshold:
        if len(max_similarity_indices) > 0:
            cluster_name = f"{cluster_names[max_similarity_indices[0]]}"
            print(f"加入的聚类名称: {cluster_name}")
            new_file_name = os.path.join("cluster", cluster_name, test_file_name)
            shutil.copyfile(test_image_path, new_file_name)
            print(f"图片 {test_file_name} 加入了聚类：{cluster_name}\n")
            print()

        else:
            model = test_file_name.split("_")[0]
            cluster_name = f"{model}"
            cluster_indices = [models.index(model) + 1]
            cluster_names = [cluster_name]
            new_file_name = os.path.join("cluster", cluster_name, test_file_name)
            shutil.copyfile(test_image_path, new_file_name)
        print()

    else:
        model = test_file_name.split("_")[0]
        cluster_name = f"{test_file_name.split('_')[0]}"
        os.makedirs(os.path.join("cluster", cluster_name), exist_ok=True)
        new_file_name = os.path.join("cluster", cluster_name, test_file_name)
        shutil.copyfile(test_image_path, new_file_name)
        print(f"图片 {test_file_name} 创建了新的聚类：{cluster_name}\n")

'''
    kmeans = KMeans(n_clusters=len(set(cluster_names)))
    cluster_features_2d = np.reshape(cluster_features, (len(cluster_features), -1))
    cluster_features_2d = np.reshape(cluster_features, (len(cluster_features), -1))
    kmeans.fit(cluster_features_2d)
    labels = kmeans.labels_
    # 计算聚类结果的轮廓系数
    silhouette_avg = silhouette_score(cluster_features_2d, labels)
    print(f"聚类结果的平均轮廓系数: {silhouette_avg}")
    '''



# 测试集文件夹路径
val_folder_path = "val_picture_contour"

# 获取验证集图片文件名
val_file_names = [file for file in os.listdir(val_folder_path) if file.endswith(".jpg")]

# 用于存储预测的聚类结果和真实类别
predicted_labels = []
true_labels = []
picture_names = []
# 对验证集图片进行预测和计算准确率
for val_file_name in val_file_names:
    val_image_path = os.path.join(val_folder_path, val_file_name)
    val_features = extract_features(val_image_path)  # 使用之前定义的特征提取函数

    # 计算验证集图片与所有聚类的相似度
    similarities = []
    for cluster_feature in cluster_features:
        similarity_score = np.mean(cosine_similarity([val_features], cluster_feature))
        similarities.append(similarity_score)

    # 找到相似度最大的聚类
    predicted_label = cluster_names[np.argmax(similarities)]

    # 将预测结果和真实类别添加到列表中
    predicted_labels.append(predicted_label)
    true_label = val_file_name.split("_")[0]
    true_labels.append(true_label)
    picture_names.append(val_file_name)
for picture_name,predicted_label , true_label in zip(picture_names,predicted_labels,true_labels):
    print("图片：" , picture_name,"预测标签为：",predicted_label,"真实标签为：",true_label)
    print()
# 计算准确率
accuracy = accuracy_score(true_labels, predicted_labels)

print("聚类模型在验证集上的准确率:", accuracy)

