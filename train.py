import argparse
import cv2
import numpy as np
import os
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from utils import get_files, get_label_from_path
from utils import read_image, load_model
from feature_extraction import get_descriptors
from feature_extraction import create_bov, extract_feature
from utils import plot_histogram, plot_confusion_matrix
from utils import write_metrics, plot_confusion_matrix


def train(
        model_clf,
        img_dir,
        extractor,
        n_visuals,
        label2idx,
        bov_path,
        **kwargs):
    img_paths = get_files(img_dir)
    image_count = len(img_paths)
    # get list descriptors from train image
    descriptor_list = []
    train_labels = []

    print('Get descriptors...')
    for img_path in img_paths:
        class_idx = get_label_from_path(img_path, label2idx)
        train_labels.append(class_idx)
        img = read_image(img_path)
        des = get_descriptors(extractor, img)
        descriptor_list.append(des)

    train_labels = np.array(train_labels)

    # stack all descriptors to np.array
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    print('Create or get bags of visual...')
    # load bov or create bov
    if os.path.exists(bov_path):
        bov = load_model(bov_path)
    else:
        bov = create_bov(descriptors, n_visuals, bov_path)

    print('Create features...')
    # create feature form bov
    im_features = extract_feature(bov, descriptor_list, image_count, n_visuals)
    # normalize feature

    scale = StandardScaler()

    print('Normalize...')
    scale.fit(im_features)
    im_features = scale.transform(im_features)

    # plot histogram
    plot_histogram(im_features, n_visuals)

    print('Training...')
    model_clf.fit(im_features, train_labels)
    print('Done')
    return model_clf, scale


def evaluate(model_clf,
             bov_path,
             img_dir,
             extractor,
             scale,
             n_visuals,
             label2idx,
             result_path):
    img_paths = get_files(img_dir)
    image_count = len(img_paths)
    # get list descriptors from train image
    descriptor_list = []
    test_labels = []

    print(f'Load bov model from {bov_path}.')
    bov = load_model(bov_path)

    print('Get descriptors')
    for img_path in img_paths:
        class_idx = get_label_from_path(img_path, label2idx)
        test_labels.append(class_idx)
        img = read_image(img_path)
        des = get_descriptors(extractor, img)
        descriptor_list.append(des)

    test_labels = np.array(test_labels)

    print('Extract feature')
    test_features = extract_feature(bov, descriptor_list, image_count, n_visuals)
    test_features = scale.transform(test_features)

    print('Predict:')
    predictions = model_clf.predict(test_features)

    labels = list(label2idx.keys())

    write_metrics(test_labels, predictions, average='macro', result_path=result_path, show=True)

    plot_confusion_matrix(test_labels, predictions, labels, normalize=True, save_dir=result_path)


if __name__ == '__main__':
    train_dir = 'data/train'
    test_dir = 'data/test'
    bov_path = 'models/bov.pkl'
    result_path = 'results'

    if os.path.exists('models') is False:
        os.makedirs('models')

    if os.path.exists(result_path) is False:
        os.makedirs(result_path)

    extractor = cv2.SIFT_create()
    model_clf = LinearSVC()
    label2idx = {
        'city': 0,
        'face': 1,
        'green': 2,
        'house_building': 3,
        'house_indoor': 4,
        'office': 5,
        'sea': 6
    }

    n_visuals = 100
    print('-'*40 + 'Training' + '-'*40)
    model_clf, scale = train(
        model_clf=model_clf,
        img_dir=train_dir,
        bov_path=bov_path,
        extractor=extractor,
        label2idx=label2idx,
        n_visuals=n_visuals
    )
    print('-' * 40 + 'Testing' + '-' * 40)
    evaluate(
        model_clf=model_clf,
        img_dir=test_dir,
        label2idx=label2idx,
        n_visuals=n_visuals,
        extractor=extractor,
        scale=scale,
        bov_path=bov_path,
        result_path=result_path
    )