from PIL import Image
import os
import pickle
import numpy as np
import pandas as pd
from skimage.measure import block_reduce
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from sklearn.cluster import Birch
from tqdm import tqdm

""" 
    image: H, W, 1
    feature: 1, 256, 64, 64
    uncertinaty: H, W
"""


class Sample_Selector(object):
    def __init__(self, csv_path: str, select_ratio: float = 0.01):
        self.csv_path = csv_path
        self.files = pd.read_csv(csv_path)["image_pth"].tolist()
        self.suffix = os.path.splitext(self.files[0])[1]

        self.select_num = int(len(self.files) * select_ratio)
        self.features, self.name_list = self._load_features()

    @staticmethod
    def average_pooling(
        uncertainty: np.ndarray, target_size: tuple = (16, 16)
    ) -> np.ndarray:
        """
        Perform average pooling to resize the uncertainty map to the target size.

        Args:
            uncertainty (np.ndarray): Input uncertainty map of shape (H, W).
            target_size (tuple): Target size (height, width), default is (64, 64).

        Returns:
            np.ndarray: Resized uncertainty map of shape (64, 64).
        """
        input_height, input_width = uncertainty.shape
        target_height, target_width = target_size

        # Calculate pooling block size
        block_size = (input_height // target_height, input_width // target_width)

        # Perform average pooling using block_reduce
        pooled_uncertainty = block_reduce(
            uncertainty, block_size=block_size, func=np.mean
        )

        return pooled_uncertainty

    @staticmethod
    def weighted_feature_fusion(
        feature: np.ndarray, pooled_uncertainty: np.ndarray
    ) -> np.ndarray:
        """
        Perform weighted fusion of feature maps using pooled uncertainty.

        Args:
            feature (np.ndarray): Input feature map of shape (1, 256, 64, 64).
            pooled_uncertainty (np.ndarray): Pooled uncertainty map of shape (64, 64).

        Returns:
            np.ndarray: Weighted feature of shape (1, 256).
        """
        # Normalize pooled_uncertainty to sum to 1
        weights = pooled_uncertainty / np.sum(pooled_uncertainty)

        # Reshape weights to match feature dimensions
        # print(feature.shape, weights.shape)
        weights = weights.reshape(1, 1, weights.shape[0], weights.shape[1])

        # Perform weighted sum along spatial dimensions
        weighted_feature = np.sum(feature * weights, axis=(2, 3))  # Shape: (1, 256)

        return weighted_feature

    @staticmethod
    def soft_numpy(array: np.ndarray, scale: int = 5) -> np.ndarray:
        """
        Apply softmax to a 2D numpy array.

        Args:
            array (np.ndarray): Input 2D array.

        Returns:
            np.ndarray: Softmax-transformed 2D array.
        """
        # Subtract the maximum value for numerical stability
        exp_array = np.exp(array * scale)
        # Compute softmax
        softmax_array = exp_array / np.sum(exp_array)
        return softmax_array

    def random_select(self, seed=2025):
        np.random.seed(seed)
        results = np.random.choice(self.files, self.select_num)
        selected_samples = [
            {
                "image_pth": image_pth,
                "mask_pth": image_pth.replace("image", "label"),
            }
            for image_pth in results
        ]

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"random_select_{self.select_num}_seed{seed}.csv",
        )
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with random method!"
        )

    def sugfw_select(self, soft_scale: int = 5, un: int = 10):

        results = []  # To store weighted_feature and avg_uncertainty for each file

        for image_file in tqdm(self.files):
            feature_file = image_file.replace(".pkl", "_feature.pkl").replace(
                "image", "feature"
            )
            uncertianty_file = image_file.replace(
                ".pkl", f"_uncertainty_{un}.pkl"
            ).replace("image", "uncertainty")
            feature, uncertainty = (
                pickle.load(open(feature_file, "rb")),
                pickle.load(open(uncertianty_file, "rb")),
            )

            pooled_uncertainty = self.average_pooling(
                uncertainty, target_size=(feature.shape[-1], feature.shape[-1])
            )
            avg_uncertainty = uncertainty.mean()
            soft_uncertainty = self.soft_numpy(pooled_uncertainty, scale=soft_scale)

            weighted_feature = self.weighted_feature_fusion(feature, soft_uncertainty)

            results.append(
                {
                    "image_pth": image_file,
                    "weighted_feature": weighted_feature,
                    "avg_uncertainty": avg_uncertainty,
                }
            )

        # Prepare data for clustering
        weighted_features = np.array(
            [result["weighted_feature"].flatten() for result in results]
        )  # Shape: (num_files, 256)

        # Perform KMeans clustering
        n_clusters = self.select_num
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        clusters = kmeans.fit_predict(weighted_features)
        # Add cluster labels to results
        for i, result in enumerate(results):
            result["cluster"] = int(clusters[i])

        # Select one sample from each cluster
        selected_samples = []  # To store selected samples
        for cluster_id in range(n_clusters):
            cluster_samples = [
                result for result in results if result["cluster"] == cluster_id
            ]

            if not selected_samples:
                # For the first cluster, select the sample with avg_uncertainty closest to the median
                uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in cluster_samples]
                )
                median_uncertainty = np.median(uncertainties)
                closest_sample = cluster_samples[
                    np.argmin(np.abs(uncertainties - median_uncertainty))
                ]
            else:
                # For subsequent clusters, select the sample with avg_uncertainty farthest from already selected samples
                selected_uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in selected_samples]
                )
                distances = np.array(
                    [
                        np.min(
                            np.abs(selected_uncertainties - sample["avg_uncertainty"])
                        )
                        for sample in cluster_samples
                    ]
                )
                closest_sample = cluster_samples[np.argmax(distances)]

            selected_samples.append(
                {
                    "image_pth": closest_sample["image_pth"],
                    "mask_pth": closest_sample["image_pth"].replace("image", "label"),
                    "avg_uncertainty": closest_sample["avg_uncertainty"],
                }
            )

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"my_select_{n_clusters}_scale{soft_scale}_un{un}.csv",
        )
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with my method!"
        )

    def sugfw_wo_featureweighting_select(self):
        results = []  # To store weighted_feature and avg_uncertainty for each file

        for image_file in tqdm(self.files):
            feature_file = image_file.replace(".pkl", "_feature.pkl").replace(
                "image", "feature"
            )
            uncertianty_file = image_file.replace(
                ".pkl", "_uncertainty_10.pkl"
            ).replace("image", "uncertainty")
            feature, uncertainty = (
                pickle.load(open(feature_file, "rb")),
                pickle.load(open(uncertianty_file, "rb")),
            )

            weighted_feature = feature.mean(axis=(2, 3))

            avg_uncertainty = uncertainty.mean()
            results.append(
                {
                    "image_pth": image_file,
                    "weighted_feature": weighted_feature,
                    "avg_uncertainty": avg_uncertainty,
                }
            )

        # Prepare data for clustering
        weighted_features = np.array(
            [result["weighted_feature"].flatten() for result in results]
        )  # Shape: (num_files, 256)

        # Perform KMeans clustering
        n_clusters = self.select_num
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        clusters = kmeans.fit_predict(weighted_features)
        # Add cluster labels to results
        for i, result in enumerate(results):
            result["cluster"] = int(clusters[i])

        # Select one sample from each cluster
        selected_samples = []  # To store selected samples
        for cluster_id in range(n_clusters):
            cluster_samples = [
                result for result in results if result["cluster"] == cluster_id
            ]

            if not selected_samples:
                # For the first cluster, select the sample with avg_uncertainty closest to the median
                uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in cluster_samples]
                )
                median_uncertainty = np.median(uncertainties)
                closest_sample = cluster_samples[
                    np.argmin(np.abs(uncertainties - median_uncertainty))
                ]
            else:
                # For subsequent clusters, select the sample with avg_uncertainty farthest from already selected samples
                selected_uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in selected_samples]
                )
                distances = np.array(
                    [
                        np.min(
                            np.abs(selected_uncertainties - sample["avg_uncertainty"])
                        )
                        for sample in cluster_samples
                    ]
                )
                closest_sample = cluster_samples[np.argmax(distances)]

            selected_samples.append(
                {
                    "image_pth": closest_sample["image_pth"],
                    "mask_pth": closest_sample["image_pth"].replace("image", "label"),
                    "avg_uncertainty": closest_sample["avg_uncertainty"],
                }
            )

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"my_select_{n_clusters}_wo_featureweighting.csv",
        )
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with my method!"
        )

    def sugfw_wo_GS_select(self, soft_scale: int = 5):

        results = []  # To store weighted_feature and avg_uncertainty for each file

        for image_file in tqdm(self.files):
            feature_file = image_file.replace(".pkl", "_feature.pkl").replace(
                "image", "feature"
            )
            uncertianty_file = image_file.replace(
                ".pkl", "_uncertainty_10.pkl"
            ).replace("image", "uncertainty")
            feature, uncertainty = (
                pickle.load(open(feature_file, "rb")),
                pickle.load(open(uncertianty_file, "rb")),
            )

            pooled_uncertainty = self.average_pooling(
                uncertainty, target_size=(feature.shape[-1], feature.shape[-1])
            )
            avg_uncertainty = uncertainty.mean()
            soft_uncertainty = self.soft_numpy(pooled_uncertainty, scale=soft_scale)

            weighted_feature = self.weighted_feature_fusion(feature, soft_uncertainty)

            results.append(
                {
                    "image_pth": image_file,
                    "weighted_feature": weighted_feature,
                    "avg_uncertainty": avg_uncertainty,
                }
            )

        # Prepare data for clustering
        weighted_features = np.array(
            [result["weighted_feature"].flatten() for result in results]
        )  # Shape: (num_files, 256)

        # Perform KMeans clustering
        n_clusters = self.select_num
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        clusters = kmeans.fit_predict(weighted_features)
        # Add cluster labels to results
        for i, result in enumerate(results):
            result["cluster"] = int(clusters[i])

        # Select one sample from each cluster
        selected_samples = []  # To store selected samples
        for cluster_id in range(n_clusters):
            cluster_samples = [
                result for result in results if result["cluster"] == cluster_id
            ]

            choosed_sample = np.random.choice(cluster_samples)

            selected_samples.append(
                {
                    "image_pth": choosed_sample["image_pth"],
                    "mask_pth": choosed_sample["image_pth"].replace("image", "label"),
                    "avg_uncertainty": choosed_sample["avg_uncertainty"],
                }
            )

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"my_select_{n_clusters}_wo_GS.csv",
        )
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with my method!"
        )

    def sugfw_wo_Clustering_select(self):

        results = []  # To store weighted_feature and avg_uncertainty for each file

        for image_file in tqdm(self.files):
            
            uncertianty_file = image_file.replace(".pkl", "_uncertainty_10.pkl").replace(
                "image", "uncertainty"
            )
            uncertainty = ( pickle.load(open(uncertianty_file, "rb")))

            avg_uncertainty = uncertainty.mean()

            results.append(
                {
                    "image_pth": image_file,
                    "avg_uncertainty": avg_uncertainty,
                }
            )

        selected_samples = []  # To store selected samples

        results.sort(key=lambda x: x["avg_uncertainty"])

        # 2. 计算总样本数
        num_results = len(results)
        
        # 3. 均匀挑选样本
        for i in range(self.select_num):
            # 计算均匀分布的索引 idx
            # 使用 (num_results - 1) * i / (self.select_num - 1) 可以覆盖到首尾样本
            # 如果 self.select_num 为 1，则直接取中间值
            if self.select_num > 1:
                idx = int(round((num_results - 1) * i / (self.select_num - 1)))
            else:
                idx = num_results // 2
            
            choosed_sample = results[idx]

            selected_samples.append(
                {
                    "image_pth": choosed_sample["image_pth"],
                    "mask_pth": choosed_sample["image_pth"].replace("image", "label"),
                    "avg_uncertainty": choosed_sample["avg_uncertainty"],
                }
            )

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"my_select_{self.select_num}_wo_Clustering.csv",
        )
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with my method!"
        )

    def fps_select(self, feats, name_list, num_samples):
        """
        Select samples using FPS (Farthest Point Sampling).
        """

        num_clusters = num_samples // 2
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(feats)
        labels = kmeans.labels_
        select_samples_list = []
        extra_sample_needed = num_samples % 2 == 1
        for i in tqdm(range(num_clusters)):
            cluster_indices = np.where(labels == i)[0]
            cluster_feats = feats[cluster_indices]
            distances = cdist(cluster_feats, cluster_feats, metric="euclidean")
            max_dist_indices = np.unravel_index(
                np.argmax(distances), distances.shape
            )
            sample1_index = cluster_indices[max_dist_indices[0]]
            sample2_index = cluster_indices[max_dist_indices[1]]
            select_samples_list.append(name_list[sample1_index])
            select_samples_list.append(name_list[sample2_index])
        if extra_sample_needed:
            last_cluster_indices = np.where(labels == num_clusters - 1)[0]
            select_samples_list.append(name_list[last_cluster_indices[0]])
        return select_samples_list[:num_samples]


    def typiclust_select(self, feats, name_list, num_samples):
        """
        Select samples using TypiClust (Typicality-based Clustering).
        """

        def compute_typicality(cluster_feats):
            distances = cdist(cluster_feats, cluster_feats, metric="euclidean")
            avg_distances = np.mean(distances, axis=1)
            return 1 / avg_distances

        kmeans = KMeans(n_clusters=num_samples, random_state=0)
        kmeans.fit(feats)
        labels = kmeans.labels_
        select_samples_list = []
        for i in tqdm(range(num_samples)):
            cluster_indices = np.where(labels == i)[0]
            cluster_feats = feats[cluster_indices]
            typicalities = compute_typicality(cluster_feats)
            highest_typicality_index = cluster_indices[np.argmax(typicalities)]
            select_samples_list.append(name_list[highest_typicality_index])
        return select_samples_list

        
    def calr_select(self, feats, name_list, num_samples):
        """
        Select samples using CALR (Clustering and Information Density).
        """

        def compute_information_density(cluster_feats):
            similarity_matrix = cosine_similarity(cluster_feats)
            return similarity_matrix.mean(axis=1)

        threshold = 0.01
        birch = Birch(threshold=threshold, n_clusters=num_samples)
        birch.fit(feats)
        labels = birch.labels_
        select_samples_list = []
        for i in tqdm(range(num_samples)):
            cluster_indices = np.where(labels == i)[0]
            cluster_feats = feats[cluster_indices]
            if cluster_feats.shape[0] == 0:
                continue
            densities = compute_information_density(cluster_feats)
            highest_density_index = cluster_indices[np.argmax(densities)]
            select_samples_list.append(name_list[highest_density_index])
        return select_samples_list

        
    def alps_select(self, feats, name_list, num_samples):
        """
        Select samples using ALPS (Active Learning via Clustering).
        """

        kmeans = KMeans(n_clusters=num_samples, random_state=0)
        kmeans.fit(feats)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        select_samples_list = []
        for i in tqdm(range(num_samples)):
            cluster_indices = np.where(labels == i)[0]
            distances = np.linalg.norm(feats[cluster_indices] - centers[i], axis=1)
            closest_index = cluster_indices[np.argmin(distances)]
            select_samples_list.append(name_list[closest_index])
        return select_samples_list

        
    def cec_select(self, k: int = 10, alpha: int=0.1, un_times: int = 10):
        """
        Calibrated Entropy-weighted Clustering (CEC) 挑选方法。
        """
        def _compute_neighbor_uncertainty(anchor_feat, neighbor_feats, neighbor_uns, alpha: float = 1.0):
            # 1. 对应公式中的 z = f(x) / ||f(x)|| (L2 归一化)
            z = anchor_feat / (np.linalg.norm(anchor_feat) + 1e-8)
            z_i = neighbor_feats / (np.linalg.norm(neighbor_feats, axis=1, keepdims=True) + 1e-8)

            # 2. 计算 ||z - z_i||^2 (欧式距离的平方)
            dist_sq = np.sum((z - z_i)**2, axis=1)

            # 3. 计算 exp(-alpha * ||z - z_i||^2)
            weights = np.exp(-alpha * dist_sq)

            # 4. 对应公式 (6): (1/k) * sum(weights * H_hat(x_i))
            k = len(neighbor_uns)
            h_nn = (1.0 / k) * np.sum(weights * neighbor_uns)
            
            return h_nn


        all_features = []
        all_avg_uncertainties = []
        valid_files = []

        print("Step 1: Loading and Preprocessing features/uncertainty...")
        for image_file in tqdm(self.files):
            feature_file = image_file.replace(".pkl", "_feature.pkl").replace("image", "feature")
            uncertainty_file = image_file.replace(".pkl", f"_uncertainty_{un_times}.pkl").replace("image", "uncertainty")
            
            if not os.path.exists(feature_file) or not os.path.exists(uncertainty_file):
                continue

            with open(feature_file, "rb") as f:
                feat = pickle.load(f) # (1, 256, 64, 64)
            with open(uncertainty_file, "rb") as f:
                un = pickle.load(f)   # (64, 64)

            # 1. 在patch上取平均
            # 特征取平均得到 (256,)
            feat_pooled = feat.mean(axis=(0, 2, 3)) 
            # 不确定性取平均得到标量
            un_pooled = un.mean()

            all_features.append(feat_pooled)
            all_avg_uncertainties.append(un_pooled)
            valid_files.append(image_file)

        all_features = np.array(all_features)           # (num, 256)
        all_avg_uncertainties = np.array(all_avg_uncertainties) # (num,)

        print(f"Step 2: Finding {k} nearest neighbors for calibration...")
        # 2. 基于image_embedding使用k近邻找到最近样本
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(all_features)
        distances, neighbor_indices = nn.kneighbors(all_features)

        updated_uncertainty = []
        for i in range(len(all_features)):
            # 获取当前样本及其近邻的数据
            anchor_f = all_features[i]
            nb_indices = neighbor_indices[i]
            nb_features = all_features[nb_indices]
            nb_uns = all_avg_uncertainties[nb_indices]

            # 计算 H_NN(x)
            h_nn = _compute_neighbor_uncertainty(anchor_f, nb_features, nb_uns, alpha=alpha)
            
            # 对应公式 (7): U(x) = H_hat(x) + H_NN(x)
            u_x = all_avg_uncertainties[i] + h_nn
            updated_uncertainty.append(u_x)

        updated_uncertainty = np.array(updated_uncertainty)

        # 5. 特征加权：基于更新后的不确定性乘回特征
        # 将 updated_uncertainty 扩展为 (num, 1) 以便广播乘法
        weighted_features = all_features * updated_uncertainty[:, np.newaxis]

        print(f"Step 3: Clustering into {self.select_num} clusters...")
        # 6. 基于最后的feature做聚类
        n_clusters = self.select_num
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(weighted_features)
        cluster_centers = kmeans.cluster_centers_

        # 7. 每个簇挑选最靠近簇中心的样本
        selected_indices = []
        for i in range(n_clusters):
            # 找到属于该簇的所有样本索引
            idx_in_cluster = np.where(cluster_labels == i)[0]
            if len(idx_in_cluster) == 0:
                continue
            
            # 计算该簇样本到其中心的距离
            feats_in_cluster = weighted_features[idx_in_cluster]
            center = cluster_centers[i]
            distances = np.linalg.norm(feats_in_cluster - center, axis=1)
            
            # 挑选距离最近的一个
            best_idx = idx_in_cluster[np.argmin(distances)]
            selected_indices.append(best_idx)

        # 整理结果
        selected_samples = [
            {
                "image_pth": valid_files[idx],
                "mask_pth": valid_files[idx].replace("image", "label"),
                "uncertainty_val": updated_uncertainty[idx]
            }
            for idx in selected_indices
        ]

        # 保存结果
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"cec_select_{n_clusters}.csv",
        )
        pd.DataFrame(selected_samples).to_csv(output_csv_file, index=False)
        print(f"Successfully saved CEC selected samples to {output_csv_file}")


        


    def probcover_select(self, feats, name_list, num_samples, delta=None, alpha=0.95):
        """
        Select samples using ProbCover (Probabilistic Coverage).
        """

        def estimate_delta(embedding, num_classes, alpha=0.95):

            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=num_classes)
            cluster_labels = kmeans.fit_predict(embedding)

            dist_matrix = distance_matrix(embedding, embedding)

            delta_values = np.linspace(0, np.max(dist_matrix), num=100)
            best_delta = 0
            for delta in tqdm(delta_values):
                pure_balls = 0
                total_balls = 0

                for i in range(len(embedding)):
                    neighbors = np.where(dist_matrix[i] <= delta)[0]
                    if len(neighbors) > 0:
                        if np.all(cluster_labels[neighbors] == cluster_labels[i]):
                            pure_balls += 1
                        total_balls += 1

                purity = pure_balls / total_balls
                if purity >= alpha:
                    best_delta = delta
                else:
                    break

            return best_delta

        n_samples = feats.shape[0]
        if delta is None:
            num_classes = num_samples
            delta = estimate_delta(feats, num_classes, alpha)
        dist_matrix = distance_matrix(feats, feats)
        adjacency_matrix = (dist_matrix <= delta).astype(int)
        selected_indices = []
        for _ in tqdm(range(num_samples)):
            out_degrees = adjacency_matrix.sum(axis=1)
            max_out_degree_index = np.argmax(out_degrees)
            selected_indices.append(max_out_degree_index)
            covered_indices = np.where(adjacency_matrix[max_out_degree_index] > 0)[
                0
            ]
            adjacency_matrix[:, covered_indices] = 0
            adjacency_matrix[covered_indices, :] = 0
        return [name_list[idx] for idx in selected_indices]


    def _load_features(self):
        """
        Load features and corresponding names.
        """
        # Implement feature loading logic here
        features = []
        name_list = []
        for image_file in tqdm(self.files):
            feature_file = image_file.replace(".pkl", "_feature.pkl").replace(
                "image", "feature"
            )
            if not os.path.exists(feature_file):
                continue
            with open(feature_file, "rb") as f:
                feature = pickle.load(f)

            feature = feature.mean(axis=(2, 3))  # Shape: (1, 256)
            feature = feature.flatten()  # Shape: (256,)

            features.append(feature)
            name_list.append(image_file)
        features = np.array(features)  # Shape: (num_files, 256)
        print("Loaded features finish!")
        return features, name_list

    def _save_selected_samples(self, selected_samples, AL_name):
        """
        Save selected samples to a CSV file.
        """
        selected_samples_df = pd.DataFrame(
            [
                {
                    "image_pth": sample,
                    "mask_pth": sample.replace("image", "label"),
                }
                for sample in selected_samples
            ]
        )
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"{AL_name}_select_{self.select_num}.csv",
        )
        selected_samples_df.to_csv(output_csv_file, index=False)
        print(
            f"Successfully saved selected samples to {output_csv_file} with {AL_name} method!"
        )

    def select_sample(self, AL_name: str):
        support_AL_methods = ["fps", "typiclust", "calr", "alps", "probcover", "cec"]
        al_method = getattr(self, f"{AL_name}_select", None)

        if al_method:
            selected_samples = al_method(self.features, self.name_list, self.select_num)
            self._save_selected_samples(selected_samples=selected_samples, AL_name=AL_name)
        
        else:
            raise NotImplementedError(f"Selection method '{AL_name}' is not supported. Available methods are: {support_AL_methods}")





if __name__ == "__main__":
    csv_paths = [
        "../data/Promise12/Foreslices/splits/train.csv",
        # "../data/UTAH/Foreslices/splits/train.csv",
        # "../data/Liver/Foreslices/splits/train.csv",
        # "../data/ISIC/Foreslices/splits/train.csv",
        # "../data/BTCV/Foreslices/splits/train.csv",
    ]

    all_select_ratios = [
        [0.03, 0.04, 0.05],
        # [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.20, 0.25, 0.30],
        # [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02],
        # [0.0004, 0.0006, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01],
        # [0.10, 0.15, 0.20, 0.25, 0.30],
    ]


    for csv_path, select_ratios in zip(csv_paths, all_select_ratios):
        for select_ratio in select_ratios:
            selector = Sample_Selector(csv_path, select_ratio=select_ratio)
            
            """ Select for SUGFW+ """
            # selector.sugfw_select(soft_scale=3, un=10)
            # selector.sugfw_select(soft_scale=4, un=10)
            # selector.sugfw_select(soft_scale=5, un=10)
            # selector.sugfw_select(soft_scale=6, un=10)
            # selector.sugfw_select(soft_scale=7, un=10)
            
            """ Select for SUGFW """
            # selector.sugfw_select(soft_scale=1, un=10)

            """ Select for ablation study """
            # selector.sugfw_wo_Clustering_select()
            # selector.sugfw_wo_featureweighting_select()
            # selector.sugfw_wo_GS_select(soft_scale=6)

            """ Select for different AL methods """
            # selector.cec_select()
            # for method in ["fps", "typiclust", "calr", "alps", "probcover", "cec"]:
            #     selector.select_sample(AL_name=method)
            
            """ Select for random sampling """
            # for seed in range(1233, 1236):
            #     selector.random_select(seed=seed)

            """ Select for param un_times """
            # selector.sugfw_select(soft_scale=6, un=5)
            # selector.sugfw_select(soft_scale=6, un=15)
            # selector.sugfw_select(soft_scale=6, un=20)



