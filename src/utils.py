import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from src.metrics import get_metrics
import pickle
import json
from sklearn.metrics import classification_report
from collections import defaultdict, Counter
import pandas as pd


def split_data_dir(dir_path, label2idx, test_rate=0.2):
    files = get_files(dir_path)
    labels = get_label_from_path(files, label2idx)
    file_df = pd.DataFrame({"file": files, "label": labels})
    train_df, test_df = split_data_balance(file_df, label_col='label', test_rate=test_rate, shuffle=True)
    return train_df['file'].tolist(), test_df['file'].tolist()


def split_data_balance(df: pd.DataFrame, label_col='label', test_rate=0.2, shuffle=False, seed=42):
    np.random.seed(seed)
    dict_label = defaultdict(list)
    for _, row in df.iterrows():
        dict_label[row[label_col]].append(row.to_dict())

    final_test = []
    final_train = []
    for label, items in dict_label.items():
        if shuffle:
            np.random.shuffle(items)
        n = len(items)
        n_train = int((1-test_rate) * n)
        final_train.extend(items[:n_train])
        final_test.extend(items[n_train:])

    train_df = pd.DataFrame(final_train)
    test_df = pd.DataFrame(final_test)
    return train_df, test_df


def load_json(file):
    with open(file, 'r') as pf:
        data = json.load(pf)
        return data


def get_label_from_path(paths, label2idx):
    if isinstance(paths, list):
        labels = []
        for path in paths:
            for key in label2idx:
                if key in path:
                    labels.append(label2idx[key])
        return labels

    for key in label2idx:
        if key in paths:
            return label2idx[key]

    raise ValueError('label value is not in label2idx')


def show_image(image, label=None):
    if label is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int(image.shape[0]/2)-30, 20)
        font_scale = 1
        color = (255, 0, 255)
        thickness = 2
        image = cv2.putText(image, label, org, font,
                            font_scale, color, thickness, cv2.LINE_AA)
    window_name = label if label else 'image'
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model


def get_files(path: str, train: bool = False):
    """
    Get all file paths from data directory
    :param path: path to data directory
    :param train: if true then shuffle paths
    :return:
    """
    images = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            images.append(os.path.join(folder_path, file))

    if train:
        np.random.shuffle(images)

    return images


def read_image(path, size=(128, 128), flags=0):
    img = cv2.imread(filename=path, flags=flags)
    return cv2.resize(img, size)


def plot_confusion_matrix(
        y_true,
        y_pred,
        target_names,
        title='Confusion matrix',
        cmap=None,
        normalize=True,
        save_dir='results'):

    cm = confusion_matrix(y_true, y_pred, target_names)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    width = int(10/7*len(target_names))

    height = int(8/7*len(target_names))

    plt.figure(figsize=(width, height))
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        title = title + ' Normalize'
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    try:
        print(f"Save confusion-matrix...")
        plt.savefig((save_dir + '/{}.png'.format(title)))
    except IOError:
        print(f"Could not save file in directory: {save_dir}")

    plt.show()


def write_metrics(y_true, y_pred, label2idx=None, average='macro', result_path='results', show=False):
    if os.path.exists(result_path) is False:
        os.makedirs(result_path)

    true_label = y_true
    pred_label = y_pred
    labels = None
    if label2idx:
        idx2label = {idx: label for label, idx in label2idx.items()}
        true_label = [idx2label[y] for y in y_true]
        pred_label = [idx2label[y] for y in y_pred]
        labels = list(label2idx.keys())

    acc, precision, recall, f1_score = get_metrics(y_true, y_pred, average)
    report = classification_report(true_label, pred_label, labels, labels=labels)
    with open(os.path.join(result_path, 'metric_scores.txt'), 'w') as pf:
        pf.write(f'Accuracy : {acc}\n')
        pf.write(f'Precision: {precision}\n')
        pf.write(f'Recall   : {recall}\n')
        pf.write(f'F1-Score : {f1_score}\n')
        pf.write('Detail\n')
        pf.write(report)

    if show:
        print(f'Accuracy : {acc}\n')
        print(f'Precision: {precision}\n')
        print(f'Recall   : {recall}\n')
        print(f'F1-Score : {f1_score}\n')
        print('Detail\n')
        print(report)


def plot_histogram(im_features, num_clusters):
    x_scalar = np.arange(num_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(num_clusters)])
    plt.bar(x_scalar.tolist(), y_scalar.tolist())
    plt.xlabel('Visual word index')
    plt.ylabel('Frequency')
    plt.title('Complete Vocabulary Generated')
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()