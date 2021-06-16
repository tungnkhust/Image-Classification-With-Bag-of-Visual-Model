import argparse
import cv2
import os
import numpy as np
from src.utils import load_json, get_files
from src.utils import read_image, get_extractor
from src.feature_extraction import get_descriptors, create_bov


def build_bov(config):
    extractor_name = config['extractor_name']
    bov_dir = config['bov_dir']
    n_visuals = config['n_visuals']

    extractor = get_extractor(extractor_name)
    if extractor is None:
        raise ValueError('extractor must be not None')

    if os.path.exists(bov_dir) is False:
        os.makedirs(bov_dir)

    # list image path of dataset
    img_paths = get_files(config['data_path'])

    if config['use_extend_image']:
        extend_img_paths = get_files(config['extend_image_dir'])
        img_paths.extend(extend_img_paths)
        print("Use extend image to build bov")
        print(f'Num image extend {len(extend_img_paths)}')

    if config['use_extend_image']:
        bov_path = os.path.join(bov_dir, f'bov_{extractor_name}_{n_visuals}_extend.sav')
    else:
        bov_path = os.path.join(bov_dir, f'bov_{extractor_name}_{n_visuals}.sav')

    print('Extraction use: ', extractor_name)
    print('N_visuals: ', n_visuals)

    print('Get descriptor')
    descriptor_list = []
    total_descriptors = 0
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        img = read_image(img_path, size=config['image_size'])
        kps, des = get_descriptors(extractor, img)
        if des is not None:
            descriptor_list.append(des)
            total_descriptors += len(des)

    descriptors = np.vstack(descriptor_list)
    print(descriptors.shape)
    print("Total descriptor: ", total_descriptors)

    print('Build bags of visual...')
    create_bov(descriptors, n_visuals, bov_path=bov_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/bov_config.json')
    args = parser.parse_args()
    config = load_json(args.config_path)
    build_bov(config)

