# BTL_CV

## Set up
#### Cài đặt môi trường
- Cài đặt môi trường anaconda theo hướng dẫn tại [đây](https://docs.anaconda.com/anaconda/install/)
- Sau khi cài đặt tạo môi trường mới:
```
conda create -n name_env python==3.6.9
conda activate name_env
```
- Cài đặt các package và thư viện liên quan
```bash
pip install -r requirement.txt
```


## Use command line interface
#### Huấn luyện models
- chỉnh sửa các siêu tham số trong file config tương tứng
- run command line:
```
python run_cli.py --mode=train --config_path=configs/natural_image_config.json
```

#### Đánh giá models
- run command line:
```
python run_cli.py --mode=eval --config_path=configs/natural_image_config.json
```
or
```
python run_cli.py --mode=eval \ 
--serialization_dir=models/NeutralImage \
--test_path=data/natural_images/test_names.csv
```

#### Dự đoán
```
python run_cli.py --mode=infer \
--config_path=configs/natural_image_config.json \
--image_path='data/natural_images/car/car_0000.jpg' \
--imshow=True
```
or
```
python run_cli.py --mode=infer \
--serialization_dir=models/NeutralImage \
--image_path='data/natural_images/car/car_0000.jpg' \
--imshow=True
```

## Config hyper parameters:
```json
{
  "label2idx": {
        "airplane": 0,
        "car": 1,
        "cat": 2,
        "dog": 3,
        "flower": 4,
        "fruit": 5,
        "motorbike": 6,
        "person": 7
    },
  "extractor_name": "Feature2D.SIFT",
  "image_size": [150, 150],
  "n_visuals": 100,
  "serialization_dir": "models/NeutralImage",
  "train_path": "data/natural_images/train_names.csv",
  "test_path": "data/natural_images/test_names.csv",
  "data_path": null,
  "result_path": "results"
}
```
- label2idx: mapper: chuyển label thành index
- extractor_name: tên của bộ trích xuất đặc trưng: hiện tại hỗ trợ 2 bộ là "Feature2D.SIFT" và "Feature2D.ORB"
- image_size: size of image after read
- n_visuals: số lượng từ trọng visuals vocab = số cụm khi phân loại
- serialization_dir: folder lưu model và config
- train_path: đường dẫn tới data train, có thể là thư mục chứa hình ảnh, hoặc file chứ các đường dần của tập train
- test_path: đường dẫn tới data test, có thể là thư mục chứa hình ảnh, hoặc file chứ các đường dần của tập test
- data_path: đường dẫn tới data, nếu như tập data chưa được chia train và test
- result_path: đường dẫn tới thư mục lưu kết quả đánh giá