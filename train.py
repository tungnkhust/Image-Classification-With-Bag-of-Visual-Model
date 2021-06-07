import argparse
import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from utils import get_files, get_label_from_path
from utils import read_image, load_model
from feature_extraction import get_descriptors
from feature_extraction import create_bov, extract_feature
from utils import plot_histogram
from utils import write_metrics, plot_confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def train(
        model_clf,
        descriptor_list,
        labels,
        extractor,
        n_visuals,
        label2idx,
        bov_path,
        model_clf_path,
        scale_path,
        **kwargs):
    #img_paths = get_files(img_dir)
    #image_count = 0
    # get list descriptors from train image
    descriptor_list = descriptor_list
    train_labels = labels
    image_count = len(train_labels)
    '''
    descriptor_list = []
    train_labels = []

    total_descriptors = 0
    print('Get descriptors...')
    for img_path in img_paths:
        class_idx = get_label_from_path(img_path, label2idx)
        img = read_image(img_path)
        des = get_descriptors(extractor, img)
        if des is not None:
            image_count += 1
            train_labels.append(class_idx)
            descriptor_list.append(des)
            total_descriptors += len(des)
    train_labels = np.array(train_labels)
    '''
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
    # plot_histogram(im_features, n_visuals)

    print('Training...')
    model_clf.fit(im_features, train_labels)
    print('Done')

    pickle.dump(model_clf, open(model_clf_path, 'wb'))
    pickle.dump(scale, open(scale_path, 'wb'))
    return model_clf, scale


def evaluate(model_clf_path,
             scale_path,
             descriptor_list,
             labels,
             extractor,
             bov_path,
             n_visuals,
             label2idx,
             result_path):
    #img_paths = get_files(img_dir)
    #image_count = 0
    # get list descriptors from train image
    #descriptor_list = []
    #test_labels = []

    print(f'Load bov model from {bov_path}.')
    bov = load_model(bov_path)
    print(f'Load classification model from {model_clf_path}.')
    model_ = load_model(model_clf_path)
    scale_ = load_model(scale_path)
    print(f'Load scale from {scale_path}.')
    print('Get descriptors')
    descriptor_list = descriptor_list
    test_labels = labels
    image_count = len(test_labels)
    '''
    for img_path in img_paths:
        class_idx = get_label_from_path(img_path, label2idx)
        img = read_image(img_path)
        des = get_descriptors(extractor, img)
        if des is not None:
            image_count += 1
            test_labels.append(class_idx)
            descriptor_list.append(des)

    test_labels = np.array(test_labels)
    '''
    print('Extract feature')
    test_features = extract_feature(bov, descriptor_list, image_count, n_visuals)
    test_features = scale_.transform(test_features)

    print('Predict:')
    predictions = model_.predict(test_features)

    labels = list(label2idx.keys())

    write_metrics(test_labels, predictions, average='macro', result_path=result_path, show=True, label2idx=label2idx)

    idx2label = {idx: label for label, idx in label2idx.items()}
    true_label = [idx2label[y] for y in test_labels]
    pred_label = [idx2label[y] for y in predictions]

    plot_confusion_matrix(true_label, pred_label, labels, normalize=True, save_dir=result_path)
    plot_confusion_matrix(true_label, pred_label, labels, normalize=False, save_dir=result_path)


if __name__ == '__main__':
    train_dir = 'data/train'
    test_dir = 'data/test'
    result_path = 'results'
    data = 'data/train'
    data = 'natural_images'

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
    label2idx = {
        'airplane': 0,
        'car': 1,
        'cat': 2,
        'dog': 3,
        'flower': 4,
        'fruit': 5,
        'motorbike': 6,
        'person':7
    }

    img_paths = get_files(data)
    # get list descriptors from train image
    descriptor_list = []
    labels = []

    print('Get descriptors...')
    for img_path in img_paths:
        class_idx = get_label_from_path(img_path, label2idx)
        img = read_image(img_path)
        des = get_descriptors(extractor, img)
        if des is not None:
            labels.append(class_idx)
            descriptor_list.append(des)
    labels = np.array(labels)
    
    train_descriptor_list, test_descriptor_list, train_labels, test_labels = train_test_split(descriptor_list, labels, test_size=0.2, random_state=41)
    n_visuals = 100
    bov_path = f'models/bov_{n_visuals}.sav'
    model_clf_path = f'models/model_clf_{n_visuals}.sav'
    scale_path = f'models/scale_{n_visuals}.sav'
    print('-'*40 + 'Training' + '-'*40)
    model_clf, scale = train(
        model_clf=model_clf,
        descriptor_list=train_descriptor_list,
        labels=train_labels,
        bov_path=bov_path,
        extractor=extractor,
        label2idx=label2idx,
        n_visuals=n_visuals,
        model_clf_path=model_clf_path,
        scale_path=scale_path
    )
    print('-' * 40 + 'Testing' + '-' * 40)

    evaluate(
        model_clf_path=model_clf_path,
        scale_path=scale_path,
        descriptor_list=test_descriptor_list,
        labels=test_labels,
        label2idx=label2idx,
        n_visuals=n_visuals,
        extractor=extractor,
        bov_path=bov_path,
        result_path=result_path
    )

