# preprocessor.py

import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ThreadPoolExecutor

class SVMPreprocessor:
    def __init__(self, image_size=64, orb_features=300, k=30):
        self.image_size = image_size
        self.orb_features = orb_features
        self.k = k
        self.scaler = StandardScaler()
        self.kmeans = None

    def load_images(self, folder):
        images, labels = [], []
        for label_name in ["cats", "dogs"]:
            path = os.path.join(folder, label_name)
            label = 0 if label_name == "cats" else 1
            if not os.path.exists(path):
                print(f"Folder not found: {path}")
                continue
            files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for f in tqdm(files, desc=f"Loading {label_name} images"):
                img_path = os.path.join(path, f)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (self.image_size, self.image_size))
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels)

    def extract_features_parallel(self, images):
        orb = cv2.ORB_create(nfeatures=self.orb_features, fastThreshold=5)

        def process(img):
            fd = hog(img, orientations=9, pixels_per_cell=(8,8),
                     cells_per_block=(2,2), block_norm='L2-Hys')
            keypoints, des = orb.detectAndCompute(img, None)
            if des is None or des.shape[0] == 0:
                des = np.array([], dtype=np.float32).reshape(0, 32)
            mean_intensity = np.mean(img) / 255.0
            edges = cv2.Canny(img, 100, 200)
            edge_density = np.sum(edges > 0) / (self.image_size ** 2)
            lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            return fd, des, mean_intensity, edge_density, hist

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process, images))

        hog_features = [r[0] for r in results]
        img_descriptors = [r[1] for r in results]
        all_descriptors = [r[1] for r in results if r[1].shape[0] > 0]
        engineered_feats = [[r[2], r[3]] for r in results]
        lbp_feats = [r[4] for r in results]

        if all_descriptors:
            all_descriptors = np.vstack(all_descriptors)
        else:
            print("Warning: No ORB descriptors found in the dataset.")
            all_descriptors = np.array([], dtype=np.float32).reshape(0, 32)

        return np.array(hog_features), img_descriptors, all_descriptors, np.array(engineered_feats), np.array(lbp_feats)

    def create_codebook(self, descriptors):
        print("Clustering descriptors with MiniBatchKMeans...")
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=500, random_state=42)
        self.kmeans.fit(descriptors)

    def compute_bow_histograms(self, img_descriptors):
        histograms = []
        for des in tqdm(img_descriptors, desc="Computing BoW histograms"):
            if des.shape[0] == 0:
                hist = np.zeros(self.k)
            else:
                labels = self.kmeans.predict(des)
                hist, _ = np.histogram(labels, bins=np.arange(self.k+1))
            histograms.append(hist)
        return np.array(histograms)

    def transform_images(self, images):
        hog_feats, img_des, _, engineered_feats, lbp_feats = self.extract_features_parallel(images)
        orb_bow = self.compute_bow_histograms(img_des)
        combined = np.hstack([hog_feats, orb_bow, engineered_feats, lbp_feats])
        return self.scaler.transform(combined)

    def fit_transform_images(self, images):
        hog_feats, img_des, all_des, engineered_feats, lbp_feats = self.extract_features_parallel(images)
        self.create_codebook(all_des)
        orb_bow = self.compute_bow_histograms(img_des)
        combined = np.hstack([hog_feats, orb_bow, engineered_feats, lbp_feats])
        return self.scaler.fit_transform(combined)
