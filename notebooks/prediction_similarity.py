import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 1. 数据准备
picture_folder = "picture_spec_contour"
image_paths = [os.path.join(picture_folder, filename) for filename in os.listdir(picture_folder)
               if os.path.isfile(os.path.join(picture_folder, filename))
               and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
# 2. 特征提取
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = base_model.predict(image)
    return features.flatten()

# 加载预训练的 VGG 模型
base_model = VGG16(weights="imagenet", include_top=False, pooling='avg')

# 提取所有图像的特征向量
#features = [extract_features(image_path) for image_path in image_paths]

# 3. 加载聚类模型
with open("cluster_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# 4. 预测测试集图像的类别并计算相似度
test_folder = "test"
test_image_paths = [os.path.join(test_folder, filename) for filename in os.listdir(test_folder)
               if os.path.isfile(os.path.join(test_folder, filename))
               and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
test_features = [extract_features(image_path) for image_path in test_image_paths]
predicted_labels = kmeans.predict(test_features)

# 输出相似度最高的三个聚类
cluster_centers = kmeans.cluster_centers_
similarities = cosine_similarity(test_features, cluster_centers)
top_k_indices = np.argsort(similarities, axis=1)[:, -3:][:, ::-1]

for i, image_path in enumerate(test_image_paths):
    print("Image:", image_path)
    print("Predicted Category:", predicted_labels[i])
    for j in top_k_indices[i]:
        cluster_label = predicted_labels[i]
        cluster_name = cluster_names.get(cluster_label, "Unknown")
        similarity = similarities[i][j]
        print(f"Cluster {j + 1} ({cluster_name}): {similarity}")
    print()
    # 保存所有相似度到文件
    with open(image_path + "_similarity.txt", "w") as f:
        for i, image_path in enumerate(test_image_paths):
            f.write("Image: " + image_path + "\n")
            f.write("Predicted Category: " + str(predicted_labels[i]) + "\n")
            for j, sim in enumerate(similarities[i]):
                f.write("Similarity to Cluster " + str(j+1) + ": " + str(sim) + "\n")
            f.write("\n")

print("Cluster labels saved to cluster_results.pkl")
print("Cluster model saved to cluster_model.pkl")
print("Image path similarities saved to image_path_similarity.txt")