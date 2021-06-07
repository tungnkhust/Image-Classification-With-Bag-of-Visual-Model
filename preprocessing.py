import os

import pandas as pd

from src.utils import split_data_balance, get_files
from src.utils import get_label_from_path


def process_natural_image_files(image_dir):
    label2idx = {
        "airplane": 0,
        "car": 1,
        "cat": 2,
        "dog": 3,
        "flower": 4,
        "fruit": 5,
        "motorbike": 6,
        "person": 7
    }
    files = get_files(image_dir)
    labels = get_label_from_path(files, label2idx)
    file_df = pd.DataFrame({"file": files, "label": labels})
    train_df, test_df = split_data_balance(file_df, label_col='label', test_rate=0.2, shuffle=True)
    files_path = os.path.join(image_dir, 'filenames.csv')
    train_path = os.path.join(image_dir, 'train_names.csv')
    test_path = os.path.join(image_dir, 'test_names.csv')
    file_df.to_csv(files_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


if __name__ == '__main__':
    image_dir = 'data/natural_images'
    process_natural_image_files(image_dir=image_dir)