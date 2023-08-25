import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

# 图像文件夹路径
image_folder = "picture_spec_contour"

# 获取所有图像文件路径
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)
               if os.path.isfile(os.path.join(image_folder, filename))
               and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# 获取图像型号列表
image_categories = []
for image_path in image_paths:
    filename = os.path.basename(image_path)
    category = filename.split("_")[0]
    image_categories.append(category)

# 定义图像特征提取函数
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

# 提取图像特征向量
features = [extract_features(image_path) for image_path in image_paths]

# 聚类模型训练
k = len(set([image_path.split("_")[0] for image_path in image_paths]))  # 聚类簇的数量为图像型号的数量
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(features)

labels = kmeans.labels_

# 4. 类别命名
# 构建图像型号到簇标签的映射
category_mapping = {}
for category, label in zip(image_categories, labels):
    if category not in category_mapping:
        category_mapping[category] = label

# 5. 保存聚类类别信息
clusters = {}
for i, label in enumerate(labels):
    image_path = image_paths[i]
    # 去掉"/"前的部分
    image_path = os.path.basename(image_path).split("/", 1)[-1]
    # 去掉第一个"_"之后的部分
    image_path = image_path.split("_", 1)[0]
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(image_path)

# 6.保存聚类结果
cluster_results = {}
for label, images in clusters.items():
    modified_images = []
    for image in images:
        # 去掉"/"前的部分
        image = os.path.basename(image).split("/", 1)[-1]
        # 去掉第一个"_"之后的部分
        image = image.split("_", 1)[0]
        modified_images.append(image)
    cluster_results[label] = modified_images
print(cluster_results)
# 保存聚类结果到文件
with open("cluster_results.pkl", "wb") as f:
    pickle.dump(cluster_results, f)

#7.保存聚类模型
with open("cluster_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# 预测图像的类别和相似度
test_folder = "picture_contour_test"
test_image_paths = [os.path.join(test_folder, filename) for filename in os.listdir(test_folder)]
test_features = [extract_features(image_path) for image_path in test_image_paths]
predicted_labels = kmeans.predict(test_features)
similarities = cosine_similarity(test_features, kmeans.cluster_centers_)

# 加载聚类模型和聚类结果
with open("cluster_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# 加载聚类结果
for i, label in enumerate(kmeans.labels_):
    if label not in cluster_results:
        cluster_results[label] = []
    cluster_results[label].append(image_paths[i])

# 更新聚类结果和模型
new_clusters = []
new_cluster_labels = []
for cluster_label, images in cluster_results.items():
    cluster_name = images[0].split("_")[0]
    for image in images:
        image_name = os.path.splitext(os.path.basename(image))[0]
        new_cluster_name = image_name.replace("_contour", "")
        if new_cluster_name == cluster_name:
            new_clusters.append(image)
            new_cluster_labels.append(cluster_label)
        else:
            new_cluster_labels.append(len(new_clusters))
            new_clusters.append(image)

# 保存更新后的聚类结果
updated_cluster_results = {}
for label, image in zip(new_cluster_labels, new_clusters):
    if label not in updated_cluster_results:
        updated_cluster_results[label] = []
    updated_cluster_results[label].append(image)

# 保存更新后的聚类结果和图片到 Excel 文件
data = {"Cluster Name": [], "Images": []}
for label, images in updated_cluster_results.items():
    cluster_name = images[0].split("_")[0]
    data["Cluster Name"].append(cluster_name)
    data["Images"].append(", ".join(images))

df = pd.DataFrame(data)
df.to_excel("model_spec.xlsx", index=False)

# 预测并保存图片相似度前三高的类别名
similarity_threshold = 0.5
results = []
for i, image_path in enumerate(test_image_paths):
    top_indices = np.argsort(similarities[i])[::-1][:3]
    top_labels = predicted_labels[top_indices]
    top_similarities = similarities[i][top_indices]
    top_names = [updated_cluster_results[label][0].split("_")[0] for label in top_labels]
    output_text = f"{os.path.splitext(os.path.basename(image_path))[0]}: "
    for name, similarity in zip(top_names, top_similarities):
        output_text += f"{name} ({similarity:.2f}), "
        if similarity > similarity_threshold:
            updated_cluster_results[top_labels[0]].append(image_path)
            break
    else:
        new_cluster_name = os.path.splitext(os.path.basename(image_path))[0].replace("_contour", "")
        new_cluster_label = len(updated_cluster_results)
        updated_cluster_results[new_cluster_label] = [image_path]
        output_text += f"New Cluster ({new_cluster_label})"

    results.append(output_text)

# 保存预测结果到文件
with open("image_predictions.txt", "w") as f:
    f.write("\n".join(results))
