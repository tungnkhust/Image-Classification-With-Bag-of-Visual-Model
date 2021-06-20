import numpy as np
import os
from sklearn.cluster import KMeans
import pickle
from src.grid import Grid
import cv2


def get_global_feature(img, global_names=('histogram', ), size=(128, 128)):
    features = []
    for global_name in sorted(set(global_names)):
        if global_name == 'histogram':
            features.append(global_histogram(img))
        if global_name == 'hog':
            features.append(global_hog(img, size))
    return np.concatenate(features, axis=-1)


def global_hog(img, size=(128, 128)):
    img = cv2.resize(img, size)
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    h = np.hstack(h)
    return h


def global_histogram(img):
    color = ('b', 'g', 'r')
    histogram = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        histogram.append(histr)
    return np.concatenate(histogram, axis=0).reshape(-1)


def get_descriptors(extractor, img):
    """
    get local feature of image
    :param extractor: extractor: suft, sift
    :param img: image
    :return: descriptors of img
    """
    # kps: key points
    # descriptors: descriptor is feature of each keypoint
    kps, descriptors = extractor.detectAndCompute(img, None)
    return kps, descriptors


def create_bov(descriptors, n_visuals, bov_path=None):
    """
    Use Kmeans algorithm to create bag of visual.
    :param descriptors: (np.array) A collection of features
    :param n_visuals: (int) Num of visual in bag of visual, n_visuals = n_clusters
    :param bov_path: path to save bov model
    :return: Kmeans model
    """
    kmeans_model = KMeans(n_clusters=n_visuals).fit(descriptors)
    if bov_path or os.path.exists(bov_path) is False:
        pickle.dump(kmeans_model, open(bov_path, 'wb'))
        print(f'Save bov model at {bov_path}')
    return kmeans_model


def extract_feature(bov, descriptor_list, n_visuals, keypoint_list=None, grid=None):
    """
    extract feature for image_count images from bov
    :param bov: (Kmeans model) bags of visual
    :param descriptor_list: (list) list descriptor of images
    :param n_visuals: (int) num of visuals of bags of visual
    :param keypoint_list: (list) list key points
    :param grid: (Grid) grid use for local extraction
    :return: (np.array) features, local_feature
    """
    # init features
    image_count = len(descriptor_list)
    im_features = np.array([np.zeros(n_visuals) for i in range(image_count)])
    local_bov_features = []
    local_features = []
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            # with each feature of image i, predict which cluster it belongs to
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, feature.shape[0])
            idx = bov.predict(feature)
            im_features[i][idx] += 1
        if grid is not None and keypoint_list is not None:
            local_bov = get_local_bov(bov, grid, keypoint_list[i], descriptor_list[i], n_visuals)
            local_bov_features.append(local_bov)
            local_features.append(get_local_features(grid, keypoint_list[i], descriptor_list[i]))

    if local_bov_features:
        local_features = np.stack(local_bov_features)

    if len(local_features) > 0:
        local_features = np.stack(local_features)
    return im_features, local_bov_features, local_features


def get_local_bov(bov, grid: Grid, key_points, descriptors, n_visuals):
    """
    Get local bov of each cel in a grid of a image.
    Each grid have WxH cell. And for each cell get bov in that cell and represented by a vector n_features dimension.
    A grid have WxH cell -> a image have WxH vector local bov, then concat them -> final local bov for each image is
    a vector with dimension is WxHxn_visuals
    :param bov: model bov
    :param grid: a grid apply to image
    :param key_points: list key points extracted by extractor
    :param descriptors: list features vector corresponding each key point
    :param n_visuals: num of visual word
    :return: a local bov
    """
    n_grid_cells = len(grid)
    local_bov_features = np.array([np.zeros(n_visuals) for i in range(n_grid_cells)])
    for i in range(len(key_points)):
        visual_id = bov.predict(descriptors[i].reshape(1, descriptors[i].shape[0]))
        cell_id = grid.get_cell_id(key_points[i])
        local_bov_features[cell_id][visual_id] += 1
    return local_bov_features.flatten()


def get_local_features(grid: Grid, key_points, descriptors):
    """
    Get local features.
    Instead of get local bov feature get local features by sum up all features vector of each keypoint
    in a cell of grid. Then concat all them of cells to a vector have dimension is WxHxn_features.
    (n_features is dimension vector output of extractor)
    :param grid: a grid apply to image
    :param key_points: list key points extracted by extractor
    :param descriptors: list features vector corresponding each key point
    :return: a local features vector
    """
    n_grid_cells = len(grid)
    n_features = descriptors[0].shape[0]
    local_features = np.array([np.zeros(n_features) for i in range(n_grid_cells)])
    for i in range(len(key_points)):
        cell_id = grid.get_cell_id(key_points[i])
        local_features[cell_id] = local_features[cell_id] + descriptors[i].reshape(1, n_features)

    return local_features
