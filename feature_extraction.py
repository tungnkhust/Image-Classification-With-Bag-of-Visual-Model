import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
from utils import get_label_from_path, get_files, read_image


def get_descriptors(extractor, img):
    """
    get local feature of image
    :param extractor: extractor: suft, sift
    :param img: image
    :return: descriptors of img
    """
    # kps: keypoints
    # descriptors: descriptor is feature of each keypoint
    kps, descriptors = extractor.detectAndCompute(img, None)
    return descriptors


def create_bov(descriptors, n_visuals, bov_path=None):
    """
    Use Kmeans algorithm to create bag of visual.
    :param descriptors: (np.array) A collection of features
    :param n_visuals: (int) Num of visual in bag of visual, n_visuals = n_clusters
    :return: Kmeans model
    """
    kmeans_model = KMeans(n_clusters=n_visuals).fit(descriptors)
    if bov_path or os.path.exists(bov_path) is False:
        pickle.dump(kmeans_model, open(bov_path, 'wb'))

    return kmeans_model


def extract_feature(bov, descriptor_list, image_count, n_visuals, n_features=128):
    """
    extract feature for image_count images from bov
    :param bov: (Kmeans model) bags of visual
    :param descriptor_list: (list) list descriptor of images
    :param image_count: (int) num of images
    :param n_visuals: (int) num of visuals of bags of visual
    :param n_features: (int) dimension of each descriptor
    :return: (np.array) features
    """
    # init features
    im_features = np.array([np.zeros(n_visuals) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            # with each feature of image i, predict which cluster it belongs to
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, feature.shape[0])
            idx = bov.predict(feature)
            im_features[i][idx] += 1

    return im_features
