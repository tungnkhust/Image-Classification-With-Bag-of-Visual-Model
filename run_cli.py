import argparse
from src.learner import BoWLearner
from src.utils import load_json
import cv2
import os


def train(config_path):
    print(f"Load config from {config_path}.")
    config = load_json(config_path)
    extractor = config['extractor_name']
    serialization_dir = config['serialization_dir']
    n_visuals = config['n_visuals']
    if extractor == 'Feature2D.SIFT':
        extractor_ = cv2.SIFT_create()
    elif extractor == 'Feature2D.ORB':
        extractor_ = cv2.ORB_create()
    else:
        extractor_ = None

    bov_path = os.path.join(serialization_dir, f'bov_{n_visuals}.sav')
    if os.path.exists(bov_path) is False:
        bov_path = None
    learner = BoWLearner(
        label2idx=config['label2idx'],
        n_visuals=n_visuals,
        image_size=config['image_size'],
        bov_path=bov_path,
        extractor=extractor_,
        serialization_dir=config['serialization_dir'],
        train_path=config['train_path'],
        test_path=config['test_path'],
        result_path=config['result_path'],
        data_path=config['data_path']
    )
    learner.train()


def evaluate(serialization_dir, test_path=None):
    learner = BoWLearner.from_serialization_dir(serialization_dir=serialization_dir)
    learner.evaluate(img_paths=test_path)


def infer(serialization_dir, image_path, imshow=False):
    learner = BoWLearner.from_serialization_dir(serialization_dir=serialization_dir)
    y_prediction = learner.predict(image_path, imshow=imshow)
    print('Prediction: ', y_prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--config_path',  type=str, default='configs/natural_image_config.json')
    parser.add_argument('--serialization_dir', type=str, default=None)
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--test_path', type=str, default='data/natural_images/test_names.csv')
    parser.add_argument('--image_path', type=str, default='data/natural_images/car/car_0000.jpg')
    parser.add_argument('--imshow', type=bool, default=False)

    args = parser.parse_args()
    if args.mode == 'train':
        train(config_path=args.config_path)
    elif args.mode == 'eval':
        if args.serialization_dir is None:
            config = load_json(args.config_path)
            print(f"Load config from {args.config_path}.")
            serialization_dir = config['serialization_dir']
            config = load_json(args.config_path)
        else:
            serialization_dir = args.serialization_dir

        if args.test_path is None:
            config = load_json(args.config_path)
            print(f"Load test_path from config {args.config_path}.")
            test_path = config['test_path']
        else:
            test_path = args.test_path

        print(args.test_path)
        evaluate(serialization_dir, test_path)

    elif args.mode == 'infer':
        if args.serialization_dir is None:
            config = load_json(args.config_path)
            print(f"Load config from {args.config_path}.")
            serialization_dir = config['serialization_dir']
        else:
            serialization_dir = args.serialization_dir
        infer(serialization_dir, image_path=args.image_path, imshow=args.imshow)