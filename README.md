# BTL_CV

## Data
Chúng tôi thực nghiệm với bộ Nature Image trên [Kaggle](https://www.kaggle.com/prasunroy/natural-images?fbclid=IwAR1nbvxfAfyQqbqTlfvr02IIGZFnrVI0oEufuL5hX0enqXrqe7HR1dFmMwA). Dữ liệu bao gồm 6899 ảnh thuộc 8 lớp khác nhau bao gồm:
- airplane: ảnh máy bay, gồm 727 ảnh.
- car: ảnh các loại xe, gồm 968 ảnh.
- cat: ảnh mèo, gồm 885 ảnh.
- dog: ảnh chó, gồm 702 ảnh.
- flower: ảnh hoa, gồm 843 ảnh.
- fruit: ảnh hoa quả, gồm 1000 ảnh.
- motorbike: ảnh xe máy, gồm 788 ảnh.
- person: ảnh người, gồm 986 ảnh

Ví dụ mẫu:

![cat](docs/img/cat_0027.jpg)

![person](docs/img/person_0106.jpg)
![dog](docs/img/dog_0051.jpg)

![car](docs/img/car_0022.jpg)
![flower](docs/img/flower_0006.jpg)
![fruit](docs/img/fruit_0004.jpg)
![airplane](docs/img/airplane_0000.jpg)
![motorbike](docs/img/motorbike_0000.jpg)


Download
Bạn có thể download trực tiếp trên [Kaggle](https://www.kaggle.com/prasunroy/natural-images?fbclid=IwAR1nbvxfAfyQqbqTlfvr02IIGZFnrVI0oEufuL5hX0enqXrqe7HR1dFmMwA) hoặc trực tiếp từ drive của chúng tôi tại [đây](https://drive.google.com/file/d/1iYSubDwyk6TFvmguWAUoalBbtVrzPh3w/view?usp=sharing).
<br> Chúng tôi có tìm kiếm thêm một số data cho các nhãn **chó**, **mèo** và **flower** bạn có thể download tại [đây](https://drive.google.com/file/d/1Gr-YTiopFQ2gXntHptjpz5rabhI4sZn6/view?usp=sharing).
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
pip install -r requirements.txt
```


## Use command line interface
#### Huấn luyện models
- chỉnh sửa các siêu tham số trong file config tương tứng
- run command line:
```
python run_cli.py --mode=train --config_path=configs/natural_image_config.json
```
Mô tả các tham số truyền:
- mode=train để lựa chọn mode huấn luyện model.
- config_path: đường dẫn tới file config.

#### Đánh giá models
- run command line:
```
python run_cli.py --mode=eval --config_path=configs/natural_image_config.json
```
or
```
python run_cli.py --mode=eval \ 
--serialization_dir=models/Local_Bov_His_Hog \
--test_path=data/natural_images/test_names.csv
```
Bạn có thể đánh giá model trực tiếp với folder lưu model sau khi huấn luyện. Hoặc có thể 
đánh gía model từ file config với các tham số truyền như sau:
- mode=eval lựa chọn mode đánh giá model
- config_path: đường dẫn tới file config. Được sử dụng khi muốn đánh giá model từ file config
- serialization_dir: đường dẫn tới folder lưu model.
- test_path: folder chứa tập ảnh test. Hoặc file chứa đường dẫn của các ảnh trong tập test.

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
--serialization_dir=models/Local_Bov_His_Hog \
--image_path='data/natural_images/car/car_0000.jpg' \
--imshow=True
```
Dự đoán phân loại của model lựa chọn.
- mode=infer lựa chọn mode dự đoán
- config_path: đường dẫn tới file config. Model sẽ được load với đường dẫn trong file config.
- serialization_dir: đường dẫn tới folder lưu model.
- image_path: đường dẫn tới ảnh muốn dự đoán
- imshow: có show hình ảnh dự đoán hay không. Muốn tắt ảnh ấn một phím bất kỳ

### Mô tả các tham số trong file config:
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
  "model_name": "MLPClassifier",
  "model_args": {},
  "image_size": [150, 150],
  "image_grid": [5, 5],
  "use_local_bov": false,
  "use_local_feature": true,
  "n_visuals": 400,
  "bov_path": "bov/bov_400.sav",
  "serialization_dir": "models/Local_Bov_His_Hog",
  "train_path": "data/natural_images/train_names.csv",
  "test_path": "data/natural_images/test_names.csv",
  "data_path": null,
  "result_path": "results/use_local_feature",
  "use_extend_image": false,
  "extend_image_dir": "data/extend",
  "use_global_feature": true,
  "global_names": ["histogram", "hog"],
  "hog_size": [128, 128]
}
```
- label2idx: dict chuyển label thành index
- extractor_name: tên của bộ trích xuất đặc trưng: hiện tại hỗ trợ 2 bộ là "Feature2D.SIFT" và "Feature2D.ORB"
- model_name: model class name
- image_size: kích thước của image khi đọc
- image_grid: kích thước của grid khi trích xuất đặc trưng cục bộ
- n_visuals: số lượng từ trọng visuals vocab bằng số cụm khi phân phân cụm
- bov_path: đường dẫn tới model bov (model phân cụm)
- serialization_dir: folder lưu model và config
- train_path: đường dẫn tới data train, có thể là thư mục chứa hình ảnh, hoặc file chứ các đường dần của tập train
- test_path: đường dẫn tới data test, có thể là thư mục chứa hình ảnh, hoặc file chứ các đường dần của tập test
- data_path: đường dẫn tới data, nếu như tập data chưa được chia train và test
- result_path: đường dẫn tới thư mục lưu kết quả đánh giá
- use_extend_image: có sử dụng thêm ảnh bên ngoài để huấn luyện hay không
- extend_image_dir: folder chứ tập ảnh thêm vào để khi huấn luyện
- use_global_feature: có sử dụng global feature hay không
- global_names: tên các global feature sử dựng, hiện tại chỉ hỗ trỡ "histogram" và "hog"
- hog_size: kích thước ảnh khi sử dụng để trích xuất đặc trưng hog.

## Build bov model
```commandline
python buil_bov.py --config_path=configs/bov_config.json
```
### Mô tả các tham số trong file config bag of visual:
```json
{
  "image_size": [150, 150],
  "extractor_name": "Feature2D.SIFT",
  "n_visuals": 100,
  "bov_dir": "models/bov",
  "data_path": "data/natural_images/train_names.csv",
  "extend_image_dir": "data/extend",
  "use_extend_image": true
}
```
- image_size: kích thước của image khi đọc
- extractor_name: tên của bộ trích xuất đặc trưng: hiện tại hỗ trợ 2 bộ là "Feature2D.SIFT" và "Feature2D.ORB"
- n_visuals: số lượng từ trọng visuals vocab bằng số cụm khi phân phân cụm
- data_path: đường dẫn tới data chứa ảnh để trích xuất các descriptor
- use_extend_image: có sử dụng thêm ảnh bên ngoài để huấn luyện hay không
- extend_image_dir: folder chứ tập ảnh thêm vào để khi huấn luyện