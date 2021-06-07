import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from metrics import get_metrics
import pickle


def get_label_from_path(path, label2idx):
    for key in label2idx:
        if key in path:
            return label2idx[key]
    raise ValueError('label value is not in label2idx')


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


def read_image(path, size=(150, 150)):
    img = cv2.imread(filename=path, flags=0)
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


def write_metrics(y_true, y_pred, average='macro', result_path='results', show=False):
    if os.path.exists(result_path) is False:
        os.makedirs(result_path)
    acc, precision, recall, f1_score, report = get_metrics(y_true, y_pred, average)
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
    y_scalar = np.array(abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(num_clusters))

    plt.bar(x_scalar, y_scalar)
    plt.xlabel('Visual word index')
    plt.ylabel('Frequency')
    plt.title('Complete Vocabulary Generated')
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()