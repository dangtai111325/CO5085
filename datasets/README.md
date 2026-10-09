# Datasets sử dụng trong CO5085

Tài liệu mô tả các tập dữ liệu của project, với thống kê từ dữ liệu local ngày
09/10/2026. Ba tập được sử dụng trong thực nghiệm là Fashion-MNIST, GTSRB và VOC2012. Dữ liệu và pretrained cache giữ trên máy; khi push Git, folder này chỉ có
`README.md`. Không đưa ảnh, nhãn, archive, NumPy arrays hoặc weights lên GitHub.

## Tổng quan

| Dataset | Notebook | Bài toán | Số lớp foreground | Train | Validation/dev | Test | Tổng ảnh dùng |
|---|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | E1, E2, E3 | Classification thời trang | 10 | 54.000 | 6.000 | 10.000 | 70.000 |
| GTSRB | A1 | Classification biển báo | 43 | 31.380 | 7.829 | 12.630 | 51.839 |
| PASCAL VOC2012 | A2 | Object detection one-stage / two-stage | 20 | 9.232 | 1.154 | 1.154 | 11.540 |

Train cập nhật weights; validation/dev chọn model, early stopping/ablation;
test đánh giá sau lựa chọn. Cả năm notebook đọc dataset của mình độc lập, không
dùng code hay output của notebook khác.

## Cấu trúc local

```text
datasets/
├── README.md                          # Public
├── fashion_mnist/                     # Chỉ local
│   ├── metadata.json
│   └── {train,val,test}/
│       ├── images.npy
│       ├── labels.npy
│       └── indices.npy
├── gtsrb/                             # Chỉ local
│   ├── metadata.json
│   └── {train,val,test}/
│       ├── images/.../*.png
│       └── samples.csv
├── voc/                               # A2, chỉ local
│   ├── {train,val,test}/               # Nguồn JPEG/XML local, có cả VOC2007
│   │   ├── images/*.jpg
│   │   ├── annotations/*.xml
│   │   └── samples.csv
│   └── voc2012/                       # Chỉ mục split VOC2012 đang dùng
│       ├── metadata.json
│       └── {train,val,test}.csv        # Tham chiếu ảnh nguồn, nhãn đọc từ XML
├── visdrone/                          # Lưu trữ local, không dùng trong notebook
│   ├── metadata.json
│   └── {train,val,test}/
│       ├── images/*.jpg
│       ├── annotations/*.txt
│       └── samples.csv
├── bdd100k/                           # Lưu trữ local, không dùng trong notebook
│   ├── verification.json
│   ├── issues.csv
│   ├── alignment_preview.jpg
│   ├── detection_cache/
│   │   └── {train,val,test}.{npz,json}
│   └── {train,val,test}/
│       ├── images/*.jpg
│       ├── annotations/*.json
│       └── samples.csv
├── pretrained/                        # Chỉ local, weights không phải ảnh train
│   ├── yolo26s.pt                     # Cache không sử dụng
│   └── checkpoints/
│       ├── deit_small_patch16_224-cd65a155.pth
│       ├── resnet50-0676ba61.pth
│       └── resnet34-b627a593.pth
└── ...                                # Cache/metadata local khác đều bị Git ignore
```

## Fashion-MNIST · E1/E2/E3

Ảnh grayscale quần áo, kích thước **28 × 28**, một nhãn integer trên mỗi ảnh.
Nguồn: [repository tác giả Zalando Research](https://github.com/zalandoresearch/fashion-mnist).
Dữ liệu nguồn có 60.000 train và 10.000 test. Local tách 600 ảnh mỗi lớp từ source
train làm validation, seed 42: train 54.000/val 6.000. Test 10.000 giữ nguyên.
Train/val không giao nhau; indices cho phép truy về source index.
Repository nguồn công bố MIT; xem [license nguồn](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

### File và nhãn

| File | Nội dung |
|---|---|
| `images.npy` | uint8 array `(N,28,28)`, pixel 0–255 |
| `labels.npy` | Integer array `(N,)`, ID 0–9 |
| `indices.npy` | Integer array `(N,)`, index trong source train hoặc source test tương ứng |
| `metadata.json` | schema, dataset, seed, source/license, split hash, normalization, count/SHA256 mỗi file |

Nhãn là vector số nguyên, không phải one-hot/CSV/JSON annotation theo ảnh.
Ví dụ ảnh i lấy bằng `images[i]`, nhãn bằng `labels[i]`.
Notebook dùng tensor float32 `(N,1,28,28)` và normalization từ train:
`(pixel/255 − 0.28588567995557596) / 0.35296029483570773`.
Không dùng test để tính mean/std. Toàn bộ ảnh được dùng; không lấy subset mặc định.

### Phân bố từng lớp

| ID | Nhãn | Train | Validation | Test |
|---:|---|---:|---:|---:|
| 0 | T-shirt/top | 5.400 | 600 | 1.000 |
| 1 | Trouser | 5.400 | 600 | 1.000 |
| 2 | Pullover | 5.400 | 600 | 1.000 |
| 3 | Dress | 5.400 | 600 | 1.000 |
| 4 | Coat | 5.400 | 600 | 1.000 |
| 5 | Sandal | 5.400 | 600 | 1.000 |
| 6 | Shirt | 5.400 | 600 | 1.000 |
| 7 | Sneaker | 5.400 | 600 | 1.000 |
| 8 | Bag | 5.400 | 600 | 1.000 |
| 9 | Ankle boot | 5.400 | 600 | 1.000 |

Tất cả 10 lớp cân bằng ở cả ba split; mỗi lớp 5.400/600/1.000 ảnh.

### Tải lại và tái tạo

Tải bốn file IDX gzip tại [data/fashion](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion):
`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`,
`t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`.
Giải mã IDX, stratify source train theo nhãn với seed 42 và 600 val/class, lưu ba
array của mỗi split; tính normalization chỉ trên train và tạo metadata.
Để tái tạo **đúng byte/fingerprint hiện tại**, dùng bản copy dataset đã xử lý kèm
metadata, vì cùng seed nhưng khác thứ tự RNG có thể cho split khác.

Fingerprint local:
`bd9f3884131e411a26892e15ff685f2849216340505ea615f3fb3b901e30fe80`.
Notebook đọc/checksum file và kiểm tra split. Có thể đặt `FASHION_MNIST_ROOT`
trỏ tới folder fashion_mnist đầy đủ ngoài project.

## GTSRB · A1

Ảnh RGB biển báo giao thông, **43 class ID 0–42**, mỗi ảnh có một nhãn.
Ảnh nguồn PPM có kích thước khác nhau; đã chuyển thành **PNG lossless** và
đối chiếu toàn bộ RGB pixels với PPM trước khi dọn bản nguồn trùng.

Nguồn tải: [kho tác giả GTSRB trên ERDA](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).
Tải `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip` và
`GTSRB_Final_Test_GT.zip`; không cần HOG/Haar/Hue features.
Giữ điều kiện sử dụng/README nguồn; trang metadata không công bố mã SPDX cho
ảnh. Trích dẫn Stallkamp, Schlipsing, Salmen và Igel.

### Chia dữ liệu

Source train 39.209 ảnh được tách thành 31.380 train và 7.829 validation.
Nhóm theo **class_id + physical_sign_track**, seed 42, khoảng 20% track mỗi class
làm validation. Không tách random từng frame của cùng biển sang hai tập.
Official test 12.630 ảnh giữ nguyên; ground-truth class lấy từ archive GT.

Đối chiếu SHA256 các PNG local: train/validation và validation/test không có ảnh
trùng; train/test có **8 ảnh trùng**, đều lớp 14 (Stop), trong track train 00014_00023.
A1 giữ full split và báo thêm accuracy/macro-F1 trên **12.622 ảnh test không trùng**.
Đây là metric độ nhạy bổ sung, không đổi dataset/manifest hoặc chọn model bằng test.

A1 resize RGB 224 × 224, ImageNet mean `[0.485,0.456,0.406]` và
std `[0.229,0.224,0.225]`. Augmentation train: rotation 10°, translation 0,05,
color jitter; không horizontal flip. Validation/test không dùng augmentation train.
ROI được giữ trong manifest làm metadata.

### CSV và đường dẫn

Mỗi split có `samples.csv`, phân cách dấu phẩy, một dòng cho một PNG.
Đường dẫn ảnh tương đối với folder split.

```csv
image,label,track,original_index,original_filename,width,height,roi_x1,roi_y1,roi_x2,roi_y2,sha256
images/00000/00000_00000.png,0,00000_00000,0,00000_00000.ppm,29,30,5,6,24,25,<SHA256-cua-PNG>
```

`label`: class integer 0–42; `track`: nhóm biển vật lý để chống leakage;
`original_index/filename`: truy nguồn; `width/height`: kích thước PNG trước resize;
`roi_*`: vùng biển từ annotation nguồn; `sha256`: checksum ảnh PNG.
Manifest test không cung cấp định danh track; trường track để rỗng.
CSV nguồn GTSRB dùng các cột Filename/Width/Height/Roi.X1…/ClassId với dấu chấm phẩy;
samples.csv local đã chuẩn hóa tên/cách phân cách như trên.

Metadata gồm schema, source, seed, group_key, counts, manifest SHA256 và split_hash.
Fingerprint local:
`c1c1759f478eeeeaac0faddf424717574f6f1895f6a5511ed77964c34eb2c7ac`.
Notebook kiểm tra manifest, label range, đủ 43 lớp, file/index không trùng và SHA256 PNG
khi đọc. Có thể đặt `GTSRB_ROOT` tới bản copy đầy đủ dataset đã xử lý.

### Phân bố class ID

A1 ánh xạ từng ClassId nguồn sang ý nghĩa biển báo; ID vẫn là 0…42.
GTSRB không cân bằng; bảng đếm **ảnh**, không phải số track.

| Class ID | Ý nghĩa | Train | Validation | Test | Tổng |
|---:|---|---:|---:|---:|---:|
| 0 | Speed limit 20 | 180 | 30 | 60 | 270 |
| 1 | Speed limit 30 | 1.770 | 450 | 720 | 2.940 |
| 2 | Speed limit 50 | 1.800 | 450 | 750 | 3.000 |
| 3 | Speed limit 60 | 1.140 | 270 | 450 | 1.860 |
| 4 | Speed limit 70 | 1.590 | 390 | 660 | 2.640 |
| 5 | Speed limit 80 | 1.500 | 360 | 630 | 2.490 |
| 6 | End speed limit 80 | 330 | 90 | 150 | 570 |
| 7 | Speed limit 100 | 1.140 | 300 | 450 | 1.890 |
| 8 | Speed limit 120 | 1.140 | 270 | 450 | 1.860 |
| 9 | No passing | 1.170 | 300 | 480 | 1.950 |
| 10 | No passing >3.5 t | 1.620 | 390 | 660 | 2.670 |
| 11 | Right-of-way intersection | 1.050 | 270 | 420 | 1.740 |
| 12 | Priority road | 1.680 | 420 | 690 | 2.790 |
| 13 | Yield | 1.740 | 420 | 720 | 2.880 |
| 14 | Stop | 630 | 150 | 270 | 1.050 |
| 15 | No vehicles | 510 | 120 | 210 | 840 |
| 16 | Vehicles >3.5 t prohibited | 330 | 90 | 150 | 570 |
| 17 | No entry | 900 | 210 | 360 | 1.470 |
| 18 | General caution | 960 | 240 | 390 | 1.590 |
| 19 | Dangerous curve left | 180 | 30 | 60 | 270 |
| 20 | Dangerous curve right | 300 | 60 | 90 | 450 |
| 21 | Double curve | 270 | 60 | 90 | 420 |
| 22 | Bumpy road | 300 | 90 | 120 | 510 |
| 23 | Slippery road | 420 | 90 | 150 | 660 |
| 24 | Road narrows right | 210 | 60 | 90 | 360 |
| 25 | Road work | 1.200 | 300 | 480 | 1.980 |
| 26 | Traffic signals | 480 | 120 | 180 | 780 |
| 27 | Pedestrians | 180 | 60 | 60 | 300 |
| 28 | Children crossing | 420 | 120 | 150 | 690 |
| 29 | Bicycle crossing | 210 | 60 | 90 | 360 |
| 30 | Ice/snow caution | 360 | 90 | 150 | 600 |
| 31 | Wild animals crossing | 630 | 150 | 270 | 1.050 |
| 32 | End speed/passing limits | 180 | 60 | 60 | 300 |
| 33 | Turn right ahead | 540 | 149 | 210 | 899 |
| 34 | Turn left ahead | 330 | 90 | 120 | 540 |
| 35 | Ahead only | 960 | 240 | 390 | 1.590 |
| 36 | Straight or right | 300 | 90 | 120 | 510 |
| 37 | Straight or left | 180 | 30 | 60 | 270 |
| 38 | Keep right | 1.650 | 420 | 690 | 2.760 |
| 39 | Keep left | 240 | 60 | 90 | 390 |
| 40 | Roundabout mandatory | 300 | 60 | 90 | 450 |
| 41 | End no passing | 180 | 60 | 60 | 300 |
| 42 | End no passing >3.5 t | 180 | 60 | 90 | 330 |

Để chuyển máy, copy gtsrb cùng metadata và ba samples.csv/ảnh. Nếu xử lý lại từ
archive nguồn, cần parse class/track, tách theo nhóm và tạo PNG/manifest/checksum;
notebook không tự tải lại các archive đã dọn.

## PASCAL VOC2012 · A2

Nguồn: [VOC2012 official](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/)
· [Detection/annotation/evaluation devkit](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html)
· [Trainval archive](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
VOC Database Rights áp dụng; ảnh có quyền của tác giả/chủ sở hữu gốc.

### Dữ liệu và cách chia tập

A2 sử dụng toàn bộ **11.540 ảnh VOC2012 trainval có nhãn**, gồm 20 lớp.
Kho JPEG/XML local có cả VOC2007; chỉ các annotation mang `folder=VOC2012`
được đưa vào thực nghiệm. Test chính thức VOC2012 không có nhãn công khai,
nên báo cáo dùng **holdout nội bộ 80/10/10**, không phải điểm trên leaderboard.

Các key được sắp xếp, xáo trộn bằng `random.Random(42)`, rồi chia thành
9.232/1.154/1.154 ảnh. Từng split được sắp theo key và lưu vào manifest.
Ba tập không giao nhau về image ID và đều có đủ 20 lớp. Manifest tham chiếu
đường dẫn tương đối tới ảnh/nhãn trong `datasets/voc`, không sao chép JPEG/XML.

| Split | Ảnh | Regular bbox | Difficult bbox | Truncated bbox (có thể giao difficult) |
|---|---:|---:|---:|---:|
| train | 9.232 | 21.975 | 3.267 | 13.182 |
| val | 1.154 | 2.788 | 468 | 1.647 |
| test | 1.154 | 2.687 | 376 | 1.561 |

### Nhãn XML và chỉ mục CSV

XML chứa `annotation/size` (width, height, depth), `object/name`,
`object/bndbox` (xmin, ymin, xmax, ymax), `difficult`, `truncated`, `pose`
và thông tin nguồn. Tên 20 lớp giữ đúng VOC: aeroplane, bicycle, bird, boat,
bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person,
pottedplant, sheep, sofa, train, tvmonitor. ID 1–20 theo thứ tự này; ID 0
là background của Faster R-CNN và ma trận chẩn đoán. RetinaNet chuyển nhãn
thành 0–19 khi tính loss, trả lại 1–20 khi suy luận.

Ví dụ XML bbox VOC 1-based inclusive:
```xml
<object>
  <name>car</name><difficult>0</difficult><truncated>1</truncated>
  <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>90</ymax></bndbox>
</object>
```
Đây là ví dụ schema. Chuyển về 0-based half-open`[9,19,110,90]`, area 101 × 71.

`voc/voc2012/{train,val,test}.csv` có các cột:
`key,image,annotation,width,height,boxes,labels,difficult,truncated`.
Bốn trường cuối là các mảng JSON trong CSV; `boxes` dùng tọa độ xyxy đã chuyển
đổi từ XML. `voc2012/metadata.json` ghi nguồn, seed, quy ước tọa độ, số ảnh/bbox
và phân bố lớp. JPEG/XML nguồn nằm trong các thư mục train/val/test của kho VOC;
việc chia tập cho A2 được quyết định bởi ba manifest VOC2012.

### Phân bố regular bbox

| Class | Train | Val | Test |
|---|---:|---:|---:|
| aeroplane | 692 | 97 | 76 |
| bicycle | 566 | 65 | 80 |
| bird | 933 | 98 | 88 |
| boat | 676 | 75 | 99 |
| bottle | 1011 | 139 | 109 |
| bus | 487 | 44 | 62 |
| car | 1624 | 189 | 204 |
| cat | 970 | 112 | 135 |
| chair | 1929 | 250 | 175 |
| cow | 473 | 48 | 67 |
| diningtable | 487 | 74 | 48 |
| dog | 1213 | 143 | 159 |
| horse | 554 | 88 | 68 |
| motorbike | 585 | 53 | 75 |
| person | 6802 | 898 | 866 |
| pottedplant | 804 | 97 | 72 |
| sheep | 611 | 119 | 83 |
| sofa | 447 | 70 | 49 |
| train | 492 | 59 | 77 |
| tvmonitor | 619 | 70 | 95 |

### DataLoader, augmentation và đánh giá

Ảnh RGB chuyển thành tensor [0,1]. Train dùng flip ngang với xác suất 0,5;
validation/test không augmentation. Model resize giữ tỷ lệ (cạnh ngắn 600,
cạnh dài tối đa 1.000), chuẩn hóa ImageNet một lần và pad batch đến bội số 32.
Batch train/eval là 4; tích lũy gradient qua 3 batch cho effective batch 12.
Không chia ảnh thành tiles. Dataset đọc chỉ mục CSV, tải ảnh khi cần và cung
cấp dữ liệu cho các bảng, đồ thị EDA.

Train học mọi bbox có nhãn, kể cả difficult. Đánh giá bỏ qua difficult và
các dự đoán lặp khớp với chúng; truncated vẫn được giữ. RetinaNet dùng
ResNet-50/FPN P3–P7, 9 anchors/vị trí, Focal Loss và Smooth L1. Faster R-CNN
dùng ResNet-50/FPN P2–P6, RPN 3 anchors/vị trí, RoI Pool 7×7 và hai FC 1.024.
Hai mạng dùng cùng seed, SGD, resize và cách chia tập. Validation VOC mAP50
chọn weights và dừng sớm: patience 2, min_delta 0,001, tối đa 30 epoch.
Cả hai lượt train/validation hoàn tất trước khi đánh giá test.

Notebook báo VOC mAP50 all-point, VOC07 AP11, 12 COCO-style AP/AR và P/R/F1
tại confidence 0,30. Có AP theo lớp, PR curves, ma trận số đếm/tỷ lệ và ảnh
GT/dự đoán/lỗi. Suy luận dùng score floor 0,05, NMS 0,5, tối đa 100 detections/ảnh.
COCO-style AP lấy 101 mốc recall, IoU 0,50–0,95; AR@1/10/100 và các nhóm
diện tích theo ngưỡng 32²/96² trên ảnh gốc. Đây là phép đo trên VOC2012,
không phải kết quả trên dataset COCO hoặc test chính thức VOC2012.

Chi phí gồm thời gian train/validation, latency median/p95, FPS batch 1,
throughput batch 4, tốc độ toàn pipeline test, peak allocated VRAM và số
tham số. Báo cáo nêu rõ workload và phần xử lý được tính trong từng phép đo.

### Tái tạo trên máy khác

Tải VOC2012 trainval từ nguồn chính thức, giải nén JPEGImages/Annotations.
Đọc XML, giữ đủ 20 lớp cùng difficult/truncated, chuyển tọa độ và tạo CSV
theo schema trên. Key local có tiền tố `2012_` trước tên file gốc để phân biệt
VOC2007; phải giữ cùng tập key và thứ tự sắp xếp trước khi shuffle seed 42.
Chia 9.232/1.154/1.154 rồi cập nhật đường dẫn tới JPEG/XML. Không cần tải test
VOC2012 không có GT. Để giữ chính xác cách chia đã dùng, sao chép ba manifest
VOC2012 và metadata cùng dữ liệu nguồn.

## VisDrone2019-DET · dữ liệu lưu trữ local

Không được notebook hiện tại sử dụng. A2 dùng VOC2012; số liệu dưới mô tả dữ liệu lưu trên máy.

Ảnh RGB từ drone, nhiều vật thể nhỏ/chen chúc và 10 lớp foreground. Nguồn:
[VisDrone của nhóm AISKYEYE, Tianjin University](https://github.com/VisDrone/VisDrone-Dataset).
Chọn **Task 1: Object Detection in Images**, không nhầm với video/tracking/counting.
Link tải train/val/test-dev trên trang nguồn; test-dev có GT công khai. Test-challenge
không có nhãn công khai nên không dùng làm test trong project.

Tải từ các link Google Drive do trang nguồn công bố:
[train ZIP](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view),
[validation ZIP](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view),
[test-dev ZIP có nhãn](https://drive.google.com/file/d/1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V/view).

### File nguồn, giải nén và chia dữ liệu

Ba ZIP được cung cấp local; ảnh JPEG và nhãn TXT được giải nén byte-for-byte sang
`visdrone/{train,val,test}/{images,annotations}`. Split test là official test-dev.
Không chia lại ảnh video thành các split ngẫu nhiên. Kiểm tra CRC từng entry,
decode RGB tất cả 8.629 JPEG, đối chiếu cùng stem với 8.629 TXT, parse 8 trường,
clamp bbox/loại box diện tích 0, ghi SHA256 ảnh/nhãn/manifest và archive nguồn.

| Split | JPEG/TXT nguồn | Ảnh dùng | Foreground GT sau lọc | Tiles 512/stride 384 |
|---|---:|---:|---:|---:|
| Train | 6.471 / 6.471 | 6.469 | 342.914 | 76.755 |
| Validation | 548 / 548 | 547 | 38.562 | 4.269 |
| Test-dev | 1.610 / 1.610 | 1.610 | 74.731 | 15.187 |
| **Tổng** | **8.629 / 8.629** | **8.626** | **456.207** | **96.211** |

Ba ảnh trùng SHA256 đều nội bộ split: train 2, val 1; giữ stem nhỏ nhất.
Không có ảnh trùng SHA256 giữa các split. File nguồn trùng vẫn giữ local;
chỉ loại khỏi manifest. `metadata.json.excluded_duplicates` liệt kê tên được giữ/loại.
Train ảnh 480–2.000 px chiều rộng/360–1.500 px chiều cao; validation và test-dev
960–1.920 px chiều rộng/540–1.080 px chiều cao. Không resize ảnh nguồn hoặc tạo
bản JPEG tiled trên đĩa.

| Archive | SHA256 |
|---|---|
| `VisDrone2019-DET-test-dev.zip` | `78b0c5078a14ee43d0b803a354e76016d7260d1704cfd1c2dc821858d839e261` |
| `VisDrone2019-DET-val.zip` | `abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9` |
| `VisDrone2019-DET-train.zip` | `86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84` |

`samples.csv` có các cột:

```csv
image,annotation,width,height,objects,image_sha256,annotation_sha256
```

Đường dẫn tương đối với split; `objects` là số GT foreground sau quy tắc bên dưới.
`metadata.json` lưu counts, phân bố nhãn/attributes, min/maxsizes, hashes, archive
provenance, ảnh loại và kết quả kiểm tra. Metadata lưu các kết quả đối chiếu manifest, TXT và JPEG khi chuẩn bị dữ liệu.
Các file này chỉ phục vụ tra cứu dữ liệu lưu trữ, không được A2 đọc.
Copy đầy đủ `visdrone` khi chuyển máy; `VISDRONE_ROOT` có thể trỏ tới bản copy này.

### Định dạng nhãn và quy tắc sử dụng

```text
bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
684,8,273,116,0,0,0,0
```

Ví dụ trên là một vùng bỏ qua (category 0/score 0), không phải background GT.
`score=1` là ứng viên đánh giá, 0 là bỏ qua. Category 1–10 là các lớp cần phát hiện;
0=ignoredregions,11=others. Truncation 0/1; occlusion 0/1/2 tương ứng không/một phần/nhiều.
[Schema và evaluation toolkit của tác giả](https://github.com/VisDrone/VisDrone2018-DET-toolkit).

Chuyển tọa độ `xywh → [x,y,x+w,y+h]` theo quy ước liên tục, clamp vào ảnh.
Loại 3 bbox diện tích 0 trong train; không có bbox phải clamp. Giữ truncated/occluded
nếu bbox hợp lệ. Dùng hợp raster của vùng category 0, loại GT có coverage ≥ 0,5:
train 252/val 157/test 371 GT. Score 0 và category 0/11 không phải foreground. Phân bố
bên dưới đã áp dụng dedup và lọc; không lấy số liệu từ predictions của model.

| Thuộc tính GT hợp lệ | Train | Validation | Test-dev |
|---|---:|---:|---:|
| Không truncation | 329.563 | 37.102 | 71.476 |
| Có truncation | 13.351 | 1.460 | 3.255 |
| Không occlusion | 167.142 | 16.723 | 41.916 |
| Occlusion một phần | 142.145 | 18.758 | 25.828 |
| Occlusion nhiều | 33.627 | 3.081 | 6.987 |

| ID | Class | Train | Validation | Test-dev |
|---:|---|---:|---:|---:|
| 1 | pedestrian | 79.251 | 8.784 | 20.755 |
| 2 | people | 27.042 | 5.109 | 6.343 |
| 3 | bicycle | 10.469 | 1.278 | 1.294 |
| 4 | car | 144.748 | 13.979 | 28.030 |
| 5 | van | 24.944 | 1.973 | 5.762 |
| 6 | truck | 12.859 | 748 | 2.653 |
| 7 | tricycle | 4.811 | 1.044 | 530 |
| 8 | awning-tricycle | 3.242 | 526 | 595 |
| 9 | bus | 5.921 | 247 | 2.939 |
| 10 | motor | 29.627 | 4.874 | 5.830 |

`people` và `pedestrian` là hai lớp riêng theo dataset. Class ID của annotation được giữ theo schema nguồn. Những nhãn này không
tham gia training hoặc evaluation của A2 trên VOC2012.

## BDD100K legacy · dữ liệu lưu trữ

Ảnh JPEG RGB cảnh lái xe, **1280 × 720** cho toàn bộ 100.000 ảnh local.
Mỗi ảnh có JSON cùng tên, nhiều object; bbox detection và polygon lane/drivable
cùng tồn tại. Thống kê detection lưu trữ chỉ dùng `box2d`, không biến polygon thành bbox.
Dataset này không được notebook hiện tại sử dụng; A2 dùng VOC2012.

Nguồn tải chính thức: [BDD100K data download](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst),
[kho download](https://dl.cv.ethz.ch/bdd100k/data/).
Chọn **100K Images** cho detection;10K Images là bộ khác.
Các version nhãn legacy và Detection 2020 khác nhau.
Xem [định dạng nguồn](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst)
và [license dataset](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst).

Bản local được tạo từ hai ZIP người dùng cung cấp, **schema legacy per-image
name/frames/objects**, có cả 20.000 test JSON. Chưa xác nhận là Detection 2020.
Checksum ZIP ảnh không khớp checksum package 100K trong tài liệu chính thức;
đã kiểm tra tính toàn vẹn/đồng bộ nội bộ, không xác nhận hai package cùng release.
Không thay nhãn bằng Detection 2020 mà vẫn dùng nguyên parser/protocol này.
Để đọc lại dữ liệu legacy này, dùng bản copy local cùng manifests/checksums.

### Nội dung, chia dữ liệu và trùng ảnh

| Split | JPEG/JSON nguồn | Ảnh sau lọc trùng | Bbox hợp lệ sau lọc ảnh và chuyển tọa độ |
|---|---:|---:|---:|
| train | 70,000 / 70,000 | 69.998 | 1.288.361 |
| val | 10,000 / 10,000 | 9.998 | 185.542 |
| test | 20,000 / 20,000 | 20.000 | 367.728 |

Số nguồn giữ nguyên: 70.000 train/10.000 val/20.000 test.
Lọc trùng theo SHA256 ảnh: ưu tiên **test > val > train**, trong cùng split giữ tên
nhỏ nhất. Có 4 ảnh bị loại khỏi danh sách dùng, gồm 2 train/2 val; raw file không bị xóa.
Không split ngẫu nhiên lại test. Val là dev; các số liệu dưới đây mô tả dữ liệu BDD lưu trữ.

| Split bị loại | Tên ảnh |
|---|---|
| train | `6e09762a-bab508de` |
| train | `abd2da13-659c6ce9` |
| val | `bdae005f-7e7e0f56` |
| val | `c8c0c00c-5a90317a` |

Không còn ảnh không có bbox sau xử lý trong split sạch; các số trong bảng nhãn
dưới đây tính trên những ảnh thực sự dùng.

### Annotation JSON

Ví dụ rút gọn từ JSON local (không mô tả polygon hoặc toàn bộ thuộc tính):

```json
{
  "name": "0000f77c-6257be58",
  "frames": [
    {
      "timestamp": 10000,
      "objects": [
        {
          "category": "traffic light",
          "id": 0,
          "attributes": {
            "occluded": false,
            "truncated": false,
            "trafficLightColor": "green"
          },
          "box2d": {
            "x1": 1125.902264,
            "y1": 133.184488,
            "x2": 1156.978645,
            "y2": 210.875445
          }
        }
      ]
    }
  ]
}
```

Ảnh tương ứng: `images/0000f77c-6257be58.jpg`; nhãn:
`annotations/0000f77c-6257be58.json`. Một frame tại timestamp 10.000 có nhiều object.
`id` là ID object trong annotation, không phải class ID.
Class lấy từ `category` và map về integer như bảng tiếp theo.
`attributes` chứa trạng thái object; `poly2d` (nếu có) không dùng cho detection.

### Ánh xạ và phân bố bbox

Background ID 0 chỉ dùng trong detector head; không phải một object được annotate.
Ánh xạ legacy: `person→pedestrian`, `motor→motorcycle`, `bike→bicycle`.
Head có 11 logits = background +10 foreground classes.

Bảng đếm **bbox**, không phải ảnh. Một ảnh có thể nhiều bbox cùng/khác lớp.
Số đo sau dedup ảnh và quy tắc bbox của cache BDD legacy.

| Detector ID | Class | Category nguồn | Train bbox | Dev bbox | Test bbox |
|---:|---|---|---:|---:|---:|
| 1 | pedestrian | person | 91.434 | 13.265 | 24.650 |
| 2 | rider | rider | 4.522 | 649 | 1.294 |
| 3 | car | car | 714.099 | 102.522 | 205.150 |
| 4 | truck | truck | 30.012 | 4.239 | 8.704 |
| 5 | bus | bus | 11.688 | 1.597 | 3.217 |
| 6 | train | train | 136 | 15 | 28 |
| 7 | motorcycle | motor | 3.002 | 452 | 841 |
| 8 | bicycle | bike | 7.227 | 1.007 | 1.998 |
| 9 | traffic light | traffic light | 186.294 | 26.887 | 52.840 |
| 10 | traffic sign | traffic sign | 239.947 | 34.909 | 69.006 |

Lớp car chiếm phần lớn bbox; train/rider/motorcycle hiếm. Không cân bằng lại bằng
cách xóa các lớp phổ biến; dùng per-class AP để đọc chất lượng mỗi lớp.

### Số ảnh chứa từng lớp

Mỗi ảnh chỉ đếm một lần cho một class, dù class đó có nhiều bbox.
Tổng qua các class có thể lớn hơn số ảnh vì đây là detection multi-label.

| Class | Train ảnh | Dev ảnh | Test ảnh |
|---|---:|---:|---:|
| pedestrian | 22.103 | 3.220 | 6.213 |
| rider | 3.590 | 515 | 1.004 |
| car | 69.199 | 9.877 | 19.776 |
| truck | 18.915 | 2.687 | 5.500 |
| bus | 9.003 | 1.242 | 2.459 |
| train | 105 | 14 | 26 |
| motorcycle | 2.284 | 334 | 640 |
| bicycle | 4.350 | 578 | 1.182 |
| traffic light | 39.278 | 5.652 | 11.051 |
| traffic sign | 57.241 | 8.219 | 16.432 |

### Tọa độ và cache detection

Theo protocol local, chuyển source inclusive `[x1,y1,x2,y2]` thành bbox liên tục
`[x1,y1,x2+1,y2+1]`, clamp về kích thước ảnh, chuyển float32 và loại box không còn
diện tích. Có 554 box source trùng endpoint trên ít nhất một trục; 553 vẫn hợp lệ
sau chuyển đổi. Loại đúng 1 box train nằm ngoài mép (x1=x2=1280), ảnh
`9f68b0a6-b2c427e6`, object ID 4. JSON gốc giữ nguyên; cache metadata ghi lý do loại.
Occluded/truncated vẫn là positive; chỉ crowd/ignored tường minh mới ignore.

`samples.csv` có các cột:

```csv
name,image,annotation,width,height,frames,boxes,polygons,image_sha256,annotation_sha256
```

Đường dẫn tương đối với split; ``name`` là stem; boxes/polygons là số object source,
không phải số bbox sau lọc. SHA256 cho phép kiểm tra JPEG/JSON đồng bộ/không đổi.

Cache `detection_cache/{train,val,test}.npz` lưu annotation đã parse từ JSON
có polygon mỗi epoch. Đây là **dữ liệu dẫn xuất local**, không phải output model;
không copy ảnh hoặc JSON sang cache.

| Array NPZ | Kiểu/shape | Vai trò |
|---|---|---|
| `boxes` | float32 `(M,4)` | Bbox đã chuyển/clamp |
| `labels` | int64 `(M,)` | Detector class ID 1–10 |
| `iscrowd` | int64 `(M,)` | Crowd flag |
| `offsets` | int64 `(N+1,)` | Bbox của ảnh i nằm trong `[offsets[i]:offsets[i+1]]` |

NPZ bám thứ tự raw samples.csv: train 70.000/val 10.000/test 20.000;
dedup ảnh diễn ra trong danh sách loader, nên **M raw cache khác tổng bbox sạch**:
train 1.288.404/val 185.578/test 367.728. JSON cạnh NPZ giữ annotation rules,
checksum arrays/source manifest và các trường hợp box bị loại. Không sửa/xóa cache
mà không sửa metadata tương ứng. A2 không đọc cache này.
Ảnh/JSON và cache legacy giữ local; không đưa vào thí nghiệm VOC2012 của A2.

### Kiểm tra nguồn local và tải lại

Đã kiểm tra CRC ZIP members, decode 100.000 JPEG, parse 100.000 JSON, name/pair,
SHA256 sau giải nén và manifests. verification.json ghi missing_images=0,
missing_annotations=0; issues.csv ghi 4 ảnh trùng và bbox có endpoint trùng.
alignment_preview.jpg là hình overlay kiểm tra từ lần chuẩn bị dữ liệu.
Những bằng chứng này giữ local, không được push.

| Archive người dùng | SHA256 ghi lại trước khi xóa ZIP |
|---|---|
| `bdd100k_images_100k.zip` | `afd149eadb49657ef5ad7bb6e9d77cc8b7927a576ee2b475b720028274e02784` |
| `bdd100k_labels.zip` | `7f1f9043c70a6ff0788a323cfb914aeced109a37b7150740d670a347c881394a` |

ZIP đã dọn sau đối chiếu với file giải nén. JPEG/JSON gốc vẫn giữ local.
Có thể đặt `BDD_ROOT` trỏ tới bản copy bdd100k đầy đủ gồm verification.json,
manifests và ba split. Để xử lý archive khác, phải kiểm tra version/schema/checksum,
match stem ảnh–nhãn, tạo samples.csv/verification và giữ cùng quy tắc dedup;
tải nhãn mới không bảo đảm tái lập label counts của bản legacy này.

## Pretrained weights sử dụng local

A1/A2 dùng trọng số ImageNet để khởi tạo backbone, không tải ảnh ImageNet
thành tập train riêng. Neck và detection head của A2 được khởi tạo mới;
không nạp trọng số detection pretrained trên COCO.

| Model | Notebook | File/nguồn weights |
|---|---|---|
| DeiT-Small/16 non-distilled | A1 | [deit_small_patch16_224-cd65a155.pth](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| ResNet-50 ImageNet-1K V1 | A1, A2 | [resnet50-0676ba61.pth](https://download.pytorch.org/models/resnet50-0676ba61.pth) |

Weights nằm trong `datasets/pretrained/checkpoints`, chỉ tải khi thiếu.
ResNet-50 có 3/4/6/3 bottleneck; C2–C5 có 256/512/1.024/2.048 channels.
A2 bỏ FC phân loại ImageNet, dùng FrozenBatchNorm và fine-tune các Conv.
Best weights chỉ giữ trong CPU RAM trong cùng phiên chạy, không ghi thêm
file checkpoint ra project.

## Dữ liệu ngoài phạm vi thực nghiệm và công bố

Các dữ liệu VOC2007, BDD100K, VisDrone, cache `pretrained/yolo26s.pt` và
`pretrained/checkpoints/resnet34-b627a593.pth` không được notebook hiện tại sử dụng.
Các file này được giữ để tra cứu trên máy, không tính vào số liệu thực nghiệm và
không đưa vào Git. verification_summary/metadata phụ cũng giữ local và bị ignore.

Quy tắc root .gitignore:

```gitignore
/datasets/*
!/datasets/README.md
```

Quy tắc này cho Git duyệt datasets nhưng chỉ cho README vào commit.
Với repo đã track dữ liệu từ trước, cần bỏ raw data khỏi index; .gitignore không
tự bỏ file đã track. Không dùng git add -f với dữ liệu/cache.
Đề bài, notebook có output cell và báo cáo được public; raw datasets giữ local.
Tôn trọng điều kiện sử dụng nguồn và trích dẫn dataset/paper trong báo cáo.
