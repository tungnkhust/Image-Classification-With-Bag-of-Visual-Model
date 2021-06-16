import json

import cv2
import numpy as np
import os
import pickle

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from src.utils import read_image, load_model, get_files, get_label_from_path, load_json
from src.feature_extraction import get_descriptors, get_local_bov
from src.feature_extraction import create_bov, extract_feature
from src.utils import write_metrics, plot_confusion_matrix, show_image
from src.data import Generator
from src.utils import split_data_balance, split_data_dir
from src.utils import get_model, get_extractor
from src.grid import Grid


class BoWLearner:
    def __init__(
            self,
            label2idx=None,
            model=None,
            extractor=None,
            scale=None,
            image_size=(128, 128),
            image_grid=(5, 5),
            use_local_bov=False,
            use_local_feature=False,
            bov=None,
            bov_path=None,
            n_visuals=50,
            serialization_dir='models',
            train_path=None,
            test_path=None,
            data_path=None,
            result_path='results',
            use_extend_image=False,
            extend_image_dir="data/extend"
    ):
        if model is None:
            model = LinearSVC()
        self.model = model

        if extractor is None:
            extractor = cv2.SIFT_create()
        self.extractor = extractor

        if scale is None:
            scale = StandardScaler()
        self.scale = scale

        if bov is None and bov_path is not None:
            bov = load_model(bov_path)
        self.bov = bov

        self.grid = Grid(width=image_grid[0], height=image_grid[1], image_size=image_size)
        self.image_size = tuple(image_size)
        self.image_grid = tuple(image_grid)
        self.use_local_bov = use_local_bov
        self.use_local_feature = use_local_feature
        self.n_visuals = n_visuals
        self.serialization_dir = serialization_dir

        if os.path.exists(serialization_dir) is False:
            os.makedirs(serialization_dir)

        self.label2idx = label2idx
        if self.label2idx is not None:
            self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        self.train_path = train_path
        self.test_path = test_path

        self.img_test_paths = None
        self.img_train_paths = None
        if train_path is None and test_path is None:
            if data_path is not None:
                if os.path.isdir(data_path):
                    self.img_train_paths, self.img_test_paths = split_data_dir(data_path, self.label2idx)
                elif os.path.isfile(data_path):
                    filenames_df = pd.read_csv(data_path)
                    train_df, test_df = split_data_balance(filenames_df, label_col='label', test_rate=0.2, shuffle=True)
                    self.img_train_paths = train_df['file'].tolist()
                    self.img_test_paths = test_df['file'].tolist()
        else:
            if train_path is not None:
                if os.path.isdir(train_path):
                    print(f'Load cho_meo files from filenames')
                    self.img_train_paths = get_files(train_path)
                else:
                    self.img_train_paths = pd.read_csv(train_path)['file'].tolist()
            else:
                if test_path is not None:
                    if os.path.isdir(test_path):
                        print(f'Load cho_meo files from filenames')
                        self.img_test_paths = get_files(test_path)
                    else:
                        self.img_test_paths = pd.read_csv(test_path)['file'].tolist()
        if use_extend_image:
            extend_img_paths = get_files(extend_image_dir)
            if self.img_train_paths is not None:
                self.img_train_paths.extend(extend_img_paths)
                print(f"Use extend image from {extend_image_dir}")
                print(f'Num of image extend {len(extend_img_paths)}')
            else:
                raise Exception('Train image paths must be not None.')
        self.result_path = result_path

    def get_im_features(self, descriptor_list, keypoint_list):
        features, local_bov_features, local_features = extract_feature(
            self.bov, descriptor_list, self.n_visuals,
            keypoint_list=keypoint_list,
            grid=self.grid
        )

        if self.use_local_bov and len(local_bov_features) > 0:
            features = np.c_[features, local_bov_features]

        if self.use_local_feature and len(local_features) > 0:
            features = np.c_[features, local_features]

        return features

    def build_bov(self, descriptor_list, bov_path=None):
        # stack all descriptors to np.array
        descriptors = np.vstack(descriptor_list)

        print('Build bags of visual...')
        # load bov or create bov
        if bov_path is None:
            bov_path = os.path.join(self.serialization_dir, f'bov_{self.n_visuals}.sav')
        self.bov = create_bov(descriptors, self.n_visuals, bov_path)

    def build_bov_from_image_generator(self, generator: Generator, bov_path=None):
        """Build bov from image generator when have large amount of image"""
        bov_model = MiniBatchKMeans(n_clusters=self.n_visuals)

        for image_batch, _ in generator:
            descriptor_list = []
            for img in image_batch:
                kps, des = get_descriptors(self.extractor, img)
                if des is not None:
                    descriptor_list.append(des)
            descriptors = np.array(descriptor_list[0])
            for descriptor in descriptor_list[1:]:
                descriptors = np.vstack((descriptors, descriptor))

            bov_model.partial_fit(descriptors)
        if bov_path is None:
            bov_path = os.path.join(self.serialization_dir, f'bov_{self.n_visuals}.sav')
        pickle.dump(bov_model, open(bov_path, 'wb'))

        self.bov = bov_model

    def get_descriptor_list(self, img_paths):
        """get all descriptors for each image of all image img_paths"""
        # get list descriptors from cho_meo image
        descriptor_list = []
        keypoint_list = []
        labels = []
        total_descriptors = 0
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            img = read_image(img_path, size=self.image_size)
            kps, des = get_descriptors(self.extractor, img)
            if des is not None:
                descriptor_list.append(des)
                keypoint_list.append(kps)
                labels.append(get_label_from_path(img_path, self.label2idx))
                total_descriptors += len(des)
        return descriptor_list, labels, total_descriptors, keypoint_list

    def get_descriptor_list_from_images(self, images):
        """get descriptor for each images"""
        image_count = 0
        # get list descriptors from cho_meo image
        descriptor_list = []
        keypoint_list = []
        total_descriptors = 0
        for img in images:
            kps, des = get_descriptors(self.extractor, img)
            if des is not None:
                image_count += 1
                descriptor_list.append(des)
                total_descriptors += len(des)
                keypoint_list.append(kps)

        return descriptor_list, keypoint_list

    def train(self, **kwargs):
        image_count = 0
        # get list descriptors from cho_meo image
        self.save_config()
        print('Create descriptor list')
        descriptor_list, train_labels, descriptor_count, keypoint_list = self.get_descriptor_list(self.img_train_paths)
        print('total descriptor in all image :', descriptor_count)
        print('total images :', len(train_labels))

        if self.bov is None:
            self.build_bov(descriptor_list)
        print('Create features...')
        # create feature form bov
        im_features = self.get_im_features(descriptor_list, keypoint_list)

        print('Normalize...')
        self.scale.fit(im_features)
        im_features = self.scale.transform(im_features)

        # plot histogram
        # plot_histogram(im_features, n_visuals)

        print('Training...')
        self.model.fit(im_features, train_labels)
        print('Done')

        # save model
        model_path = os.path.join(self.serialization_dir, f'model.sav')
        scale_path = os.path.join(self.serialization_dir, f'scale.sav')
        bov_path = os.path.join(self.serialization_dir, f'bov.sav')
        pickle.dump(self.model, open(model_path, 'wb'))
        pickle.dump(self.scale, open(scale_path, 'wb'))
        pickle.dump(self.bov, open(bov_path, 'wb'))

        if self.img_test_paths is not None:
            self.evaluate(self.img_test_paths, result_path=self.result_path)
        return self.model, self.scale

    def evaluate(self, img_paths=None, result_path='results'):
        if img_paths is not None:
            if os.path.isfile(img_paths) and os.path.exists(img_paths):
                img_paths = pd.read_csv(img_paths)['file'].tolist()
            elif os.path.isdir(img_paths) is False:
                img_paths = get_files(img_paths)
            else:
                if self.img_test_paths:
                    img_paths = self.img_test_paths
                else:
                    raise Exception('img_paths must be not None')

        # get list descriptors from cho_meo image
        descriptor_list, test_labels, _, keypoint_list = self.get_descriptor_list(img_paths)

        test_labels = np.array(test_labels)

        print('Extract feature')
        test_features = self.get_im_features(descriptor_list, keypoint_list)
        test_features = self.scale.transform(test_features)

        print('Predict:')
        predictions = self.model.predict(test_features)

        labels = list(self.label2idx.keys())

        write_metrics(test_labels, predictions, average='macro',
                      result_path=result_path,
                      show=True, label2idx=self.label2idx)

        true_label = [self.idx2label[y] for y in test_labels]
        pred_label = [self.idx2label[y] for y in predictions]

        plot_confusion_matrix(true_label, pred_label, labels, normalize=True, save_dir=result_path)
        plot_confusion_matrix(true_label, pred_label, labels, normalize=False, save_dir=result_path)

    def train_generator(
            self,
            train_generator: Generator,
            img_test_paths=None,
            test_labels=None,
            result_path='results',
            **kwargs
    ):
        image_count = 0
        # get list descriptors from cho_meo image
        train_labels = []
        im_features = []

        if self.bov is None:
            print("build bags of word from generator")
            self.build_bov_from_image_generator(train_generator)

        print('Train minibatch')
        for image_batch,  label_batch in train_generator:
            descriptor_list, keypoint_list = self.get_descriptor_list_from_images(image_batch)
            im_features, local_features = extract_feature(self.bov, descriptor_list, self.n_visuals,
                                                          keypoint_list=keypoint_list, grid=self.grid)

            print('Use local bov: ', self.use_local_bov)
            print(len(local_features))
            if self.use_local_bov and len(local_features) > 0:
                print('Use local bov')
                im_features = np.c_[im_features, local_features]

            # normalize feature
            self.scale.fit(im_features)
            im_features = self.scale.transform(im_features)

            print('Training...')
            try:
                self.model.partial_fit(im_features, train_labels)
            except:
                raise Exception('model must be incremental estimators')

                # save model
            model_path = os.path.join(self.serialization_dir, f'model.sav')
            scale_path = os.path.join(self.serialization_dir, f'scale.sav')
            bov_path = os.path.join(self.serialization_dir, f'bov.sav')
            pickle.dump(self.model, open(model_path, 'wb'))
            pickle.dump(self.scale, open(scale_path, 'wb'))
            pickle.dump(self.bov, open(bov_path, 'wb'))

        if img_test_paths is not None:
            self.evaluate(img_test_paths, result_path=result_path)
        return self.model, self.scale

    def predict(self, image_path, imshow=False):
        if isinstance(image_path, str) and os.path.exists(image_path):
            image = read_image(image_path, size=self.image_size)
        elif isinstance(image_path, np.ndarray):
            image = image_path
        else:
            raise ValueError(f'image value {image_path} error. image argument must be np.array or path to image')

        descriptor_list, keypoint_list = self.get_descriptor_list_from_images([image])
        im_features = self.get_im_features(descriptor_list, keypoint_list)

        im_features = self.scale.transform(im_features)
        y_pred = self.model.predict(im_features)[0]
        y_label = self.idx2label[y_pred]
        if imshow:
            if isinstance(image_path, str):
                image = read_image(image_path, size=self.image_size, flags=1)
            show_image(image, y_label)
        return y_label

    def save_config(self, config_path=None):
        config = dict()
        config['serialization_dir'] = self.serialization_dir
        config['label2idx'] = self.label2idx
        config['n_visuals'] = self.n_visuals
        config['image_size'] = list(self.image_size)
        config['model_name'] = self.model.__class__.__name__
        config['extractor_name'] = self.extractor.getDefaultName()
        config['image_grid'] = self.image_grid
        config['use_local_bov'] = self.use_local_bov
        config['use_local_feature'] = self.use_local_feature
        if config_path is None:
            config_path = os.path.join(self.serialization_dir, 'config.json')

        print(f"Save config at {config_path}")
        with open(config_path, 'w') as pf:
            json.dump(config, pf)

    @classmethod
    def from_serialization_dir(
            cls,
            serialization_dir,
            train_path=None,
            test_path=None,
            data_path=None
    ):
        config_path = os.path.join(serialization_dir, 'config.json')
        config = load_json(config_path)
        bov_path = os.path.join(serialization_dir, f'bov.sav')
        bov = load_model(bov_path)
        model_path = os.path.join(serialization_dir, f'model.sav')
        model = load_model(model_path)
        scale_path = os.path.join(serialization_dir, f'scale.sav')
        scale = load_model(scale_path)

        extractor = get_extractor(config['extractor_name'])

        return cls(
            label2idx=config['label2idx'],
            model=model,
            extractor=extractor,
            bov=bov,
            scale=scale,
            n_visuals=config['n_visuals'],
            image_size=config['image_size'],
            train_path=train_path,
            test_path=test_path,
            data_path=data_path,
            use_local_bov=config['use_local_bov'],
            image_grid=config['image_grid'],
            use_local_feature=config['use_local_feature']
        )

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            if os.path.exists(config):
                config = load_json(config)
            else:
                raise TypeError('config must be dict type')
        elif isinstance(config, dict) is False:
            raise TypeError('config must be dict type')

        serialization_dir = config['serialization_dir']
        n_visuals = config['n_visuals']

        if config['bov_path'] is None:
            bov_path = os.path.join(serialization_dir, f'bov.sav')
        else:
            bov_path = config['bov_path']

        if os.path.exists(bov_path):
            print(f"load bov from {bov_path}")
            bov = load_model(bov_path)
        else:
            bov = None

        model_path = os.path.join(serialization_dir, f'model.sav')
        if os.path.exists(model_path):
            print(f'load model from {model_path}')
            model = load_model(model_path)
        else:
            model = get_model(config['model_name'], **config['model_args'])

        scale_path = os.path.join(serialization_dir, f'scale.sav')
        if os.path.exists(scale_path):
            scale = load_model(scale_path)
        else:
            scale = StandardScaler()

        extractor = get_extractor(config['extractor_name'])

        return cls(
            label2idx=config['label2idx'],
            model=model,
            extractor=extractor,
            bov=bov,
            scale=scale,
            n_visuals=n_visuals,
            image_size=config['image_size'],
            train_path=config['train_path'],
            test_path=config['test_path'],
            data_path=config['data_path'],
            image_grid=config['image_grid'],
            use_local_bov=config['use_local_bov'],
            use_local_feature=config['use_local_feature'],
            serialization_dir=serialization_dir,
            extend_image_dir=config['extend_image_dir'],
            use_extend_image=config['use_extend_image']
        )
