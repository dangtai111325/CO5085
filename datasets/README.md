# Datasets sử dụng trong CO5085

Tài liệu mô tả dữ liệu **đang dùng bởi năm notebook**, thống kê lại từ file local ngày
07/10/2026. Dữ liệu và pretrained cache giữ trên máy; khi push Git, folder này chỉ có
`README.md`. Không đưa ảnh, nhãn, archive, NumPy arrays hoặc weights lên GitHub.

## Tổng quan

| Dataset | Notebook | Bài toán | Số lớp foreground | Train | Validation/dev | Test | Tổng ảnh dùng |
|---|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | E1, E2, E3 | Classification thời trang | 10 | 54.000 | 6.000 | 10.000 | 70.000 |
| GTSRB | A1 | Classification biển báo | 43 | 31.380 | 7.829 | 12.630 | 51.839 |
| BDD100K legacy local | A2 | Object detection đường phố | 10 | 69.998 | 9.998 | 20.000 | 99.996 |

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
├── bdd100k/                           # Chỉ local
│   ├── verification.json
│   ├── issues.csv
│   ├── alignment_preview.jpg
│   ├── detection_cache/
│   │   └── {train,val,test}.{npz,json}
│   └── {train,val,test}/
│       ├── images/*.jpg
│       ├── annotations/*.json
│       └── samples.csv
├── pretrained/checkpoints/            # Chỉ local, weights không phải ảnh train
│   ├── deit_small_patch16_224-cd65a155.pth
│   └── resnet50-0676ba61.pth
└── ...                                # Cache/metadata local khác đều bị Git ignore
```

## Fashion-MNIST · E1/E2/E3

Ảnh grayscale quần áo, kích thước **28×28**, một nhãn integer trên mỗi ảnh.
Nguồn: [repository tác giả Zalando Research](https://github.com/zalandoresearch/fashion-mnist).
Dữ liệu nguồn có60.000 train và10.000 test. Local tách600 ảnh mỗi lớp từ source
train làm validation, seed42: train54.000/val6.000. Test10.000 giữ nguyên.
Train/val không giao nhau; indices cho phép truy về source index.
Repository nguồn công bố MIT; xem [license nguồn](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

### File và nhãn

| File | Nội dung |
|---|---|
| `images.npy` | uint8 array `(N,28,28)`, pixel0–255 |
| `labels.npy` | Integer array `(N,)`, ID0–9 |
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
| 0 | T-shirt/top | 5,400 | 600 | 1,000 |
| 1 | Trouser | 5,400 | 600 | 1,000 |
| 2 | Pullover | 5,400 | 600 | 1,000 |
| 3 | Dress | 5,400 | 600 | 1,000 |
| 4 | Coat | 5,400 | 600 | 1,000 |
| 5 | Sandal | 5,400 | 600 | 1,000 |
| 6 | Shirt | 5,400 | 600 | 1,000 |
| 7 | Sneaker | 5,400 | 600 | 1,000 |
| 8 | Bag | 5,400 | 600 | 1,000 |
| 9 | Ankle boot | 5,400 | 600 | 1,000 |

Tất cả10 lớp cân bằng ở cả ba split; mỗi lớp5.400/600/1.000 ảnh.

### Tải lại và tái tạo

Tải bốn file IDX gzip tại [data/fashion](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion):
`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`,
`t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`.
Giải mã IDX, stratify source train theo nhãn với seed42 và600 val/class, lưu ba
array của mỗi split; tính normalization chỉ trên train và tạo metadata.
Để tái tạo **đúng byte/fingerprint hiện tại**, dùng bản copy dataset đã xử lý kèm
metadata, vì cùng seed nhưng khác thứ tự RNG có thể cho split khác.

Fingerprint local:
`bd9f3884131e411a26892e15ff685f2849216340505ea615f3fb3b901e30fe80`.
Notebook đọc/checksum file và kiểm tra split. Có thể đặt `FASHION_MNIST_ROOT`
trỏ tới folder fashion_mnist đầy đủ ngoài project.

## GTSRB · A1

Ảnh RGB biển báo giao thông, **43 class ID0–42**, mỗi ảnh có một nhãn.
Ảnh nguồn PPM có kích thước khác nhau; đã chuyển thành **PNG lossless** và
đối chiếu toàn bộ RGB pixels với PPM trước khi dọn bản nguồn trùng.

Nguồn tải: [kho tác giả GTSRB trên ERDA](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).
Tải `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip` và
`GTSRB_Final_Test_GT.zip`; không cần HOG/Haar/Hue features.
Giữ điều kiện sử dụng/README nguồn; trang metadata không công bố mã SPDX cho
ảnh. Trích dẫn Stallkamp, Schlipsing, Salmen và Igel.

### Chia dữ liệu

Source train39.209 ảnh được tách thành31.380 train và7.829 validation.
Nhóm theo **class_id + physical_sign_track**, seed42, khoảng20% track mỗi class
làm validation. Không tách random từng frame của cùng biển sang hai tập.
Official test12.630 ảnh giữ nguyên; ground-truth class lấy từ archive GT.

Đối chiếu SHA256 các PNG local: train/validation và validation/test không có ảnh
trùng; train/test có **8 ảnh trùng**, đều lớp14 (Stop), trong track train00014_00023.
A1 giữ full split và báo thêm accuracy/macro-F1 trên **12.622 ảnh test không trùng**.
Đây là metric độ nhạy bổ sung, không đổi dataset/manifest hoặc chọn model bằng test.

A1 resize RGB224×224, ImageNet mean `[0.485,0.456,0.406]` và
std `[0.229,0.224,0.225]`. Augmentation train: rotation10°, translation0,05,
color jitter; không horizontal flip. Validation/test không dùng augmentation train.
ROI được giữ trong manifest làm metadata.

### CSV và đường dẫn

Mỗi split có `samples.csv`, phân cách dấu phẩy, một dòng cho một PNG.
Đường dẫn ảnh tương đối với folder split.

```csv
image,label,track,original_index,original_filename,width,height,roi_x1,roi_y1,roi_x2,roi_y2,sha256
images/00000/00000_00000.png,0,00000_00000,0,00000_00000.ppm,29,30,5,6,24,25,<SHA256-cua-PNG>
```

`label`: class integer0–42; `track`: nhóm biển vật lý để chống leakage;
`original_index/filename`: truy nguồn; `width/height`: kích thước PNG trước resize;
`roi_*`: vùng biển từ annotation nguồn; `sha256`: checksum ảnh PNG.
Manifest test không cung cấp định danh track; trường track để rỗng.
CSV nguồn GTSRB dùng các cột Filename/Width/Height/Roi.X1…/ClassId với dấu chấm phẩy;
samples.csv local đã chuẩn hóa tên/cách phân cách như trên.

Metadata gồm schema, source, seed, group_key, counts, manifest SHA256 và split_hash.
Fingerprint local:
`c1c1759f478eeeeaac0faddf424717574f6f1895f6a5511ed77964c34eb2c7ac`.
Notebook kiểm tra manifest, label range, đủ43 lớp, file/index không trùng và SHA256 PNG
khi đọc. Có thể đặt `GTSRB_ROOT` tới bản copy đầy đủ dataset đã xử lý.

### Phân bố class ID

A1 ánh xạ từng ClassId nguồn sang ý nghĩa biển báo; ID vẫn là0…42.
GTSRB không cân bằng; bảng đếm **ảnh**, không phải số track.

| Class ID | Ý nghĩa | Train | Validation | Test | Tổng |
|---:|---|---:|---:|---:|---:|
| 0 | Speed limit 20 | 180 | 30 | 60 | 270 |
| 1 | Speed limit 30 | 1,770 | 450 | 720 | 2,940 |
| 2 | Speed limit 50 | 1,800 | 450 | 750 | 3,000 |
| 3 | Speed limit 60 | 1,140 | 270 | 450 | 1,860 |
| 4 | Speed limit 70 | 1,590 | 390 | 660 | 2,640 |
| 5 | Speed limit 80 | 1,500 | 360 | 630 | 2,490 |
| 6 | End speed limit 80 | 330 | 90 | 150 | 570 |
| 7 | Speed limit 100 | 1,140 | 300 | 450 | 1,890 |
| 8 | Speed limit 120 | 1,140 | 270 | 450 | 1,860 |
| 9 | No passing | 1,170 | 300 | 480 | 1,950 |
| 10 | No passing >3.5t | 1,620 | 390 | 660 | 2,670 |
| 11 | Right-of-way intersection | 1,050 | 270 | 420 | 1,740 |
| 12 | Priority road | 1,680 | 420 | 690 | 2,790 |
| 13 | Yield | 1,740 | 420 | 720 | 2,880 |
| 14 | Stop | 630 | 150 | 270 | 1,050 |
| 15 | No vehicles | 510 | 120 | 210 | 840 |
| 16 | Vehicles >3.5t prohibited | 330 | 90 | 150 | 570 |
| 17 | No entry | 900 | 210 | 360 | 1,470 |
| 18 | General caution | 960 | 240 | 390 | 1,590 |
| 19 | Dangerous curve left | 180 | 30 | 60 | 270 |
| 20 | Dangerous curve right | 300 | 60 | 90 | 450 |
| 21 | Double curve | 270 | 60 | 90 | 420 |
| 22 | Bumpy road | 300 | 90 | 120 | 510 |
| 23 | Slippery road | 420 | 90 | 150 | 660 |
| 24 | Road narrows right | 210 | 60 | 90 | 360 |
| 25 | Road work | 1,200 | 300 | 480 | 1,980 |
| 26 | Traffic signals | 480 | 120 | 180 | 780 |
| 27 | Pedestrians | 180 | 60 | 60 | 300 |
| 28 | Children crossing | 420 | 120 | 150 | 690 |
| 29 | Bicycle crossing | 210 | 60 | 90 | 360 |
| 30 | Ice/snow caution | 360 | 90 | 150 | 600 |
| 31 | Wild animals crossing | 630 | 150 | 270 | 1,050 |
| 32 | End speed/passing limits | 180 | 60 | 60 | 300 |
| 33 | Turn right ahead | 540 | 149 | 210 | 899 |
| 34 | Turn left ahead | 330 | 90 | 120 | 540 |
| 35 | Ahead only | 960 | 240 | 390 | 1,590 |
| 36 | Straight or right | 300 | 90 | 120 | 510 |
| 37 | Straight or left | 180 | 30 | 60 | 270 |
| 38 | Keep right | 1,650 | 420 | 690 | 2,760 |
| 39 | Keep left | 240 | 60 | 90 | 390 |
| 40 | Roundabout mandatory | 300 | 60 | 90 | 450 |
| 41 | End no passing | 180 | 60 | 60 | 300 |
| 42 | End no passing >3.5t | 180 | 60 | 90 | 330 |

Để chuyển máy, copy gtsrb cùng metadata và ba samples.csv/ảnh. Nếu xử lý lại từ
archive nguồn, cần parse class/track, tách theo nhóm và tạo PNG/manifest/checksum;
notebook không tự tải lại các archive đã dọn.

## BDD100K legacy local · A2

Ảnh JPEG RGB cảnh lái xe, **1280×720** cho toàn bộ100.000 ảnh local.
Mỗi ảnh có JSON cùng tên, nhiều object; bbox detection và polygon lane/drivable
cùng tồn tại. A2 chỉ dùng `box2d`; không biến polygon thành bbox.

Nguồn tải chính thức: [BDD100K data download](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst),
[kho download](https://dl.cv.ethz.ch/bdd100k/data/).
Chọn **100K Images** cho detection;10K Images là bộ khác.
Các version nhãn legacy và Detection2020 khác nhau.
Xem [định dạng nguồn](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst)
và [license dataset](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst).

Bản local được tạo từ hai ZIP người dùng cung cấp, **schema legacy per-image
name/frames/objects**, có cả20.000 test JSON. Chưa xác nhận là Detection2020.
Checksum ZIP ảnh không khớp checksum package100K trong tài liệu chính thức;
đã kiểm tra tính toàn vẹn/đồng bộ nội bộ, không xác nhận hai package cùng release.
Không thay nhãn bằng Detection2020 mà vẫn dùng nguyên parser/protocol này.
Nếu muốn tái lập đúng thí nghiệm hiện tại, dùng bản copy local cùng manifests/checksums.

### Nội dung, chia dữ liệu và trùng ảnh

| Split | JPEG/JSON nguồn | Ảnh sau lọc trùng | Bbox hợp lệ sau lọc ảnh và chuyển tọa độ |
|---|---:|---:|---:|
| train | 70,000 / 70,000 | 69,998 | 1,288,361 |
| val | 10,000 / 10,000 | 9,998 | 185,542 |
| test | 20,000 / 20,000 | 20,000 | 367,728 |

Số raw giữ nguyên:70.000 train/10.000 val/20.000 test.
Lọc trùng theo SHA256 ảnh: ưu tiên **test > val > train**, trong cùng split giữ tên
nhỏ nhất. Có4 ảnh bị loại khỏi danh sách dùng, gồm2train/2val; raw file không bị xóa.
Không split ngẫu nhiên lại test. Val là dev. Mỗi Run All dùng toàn bộ split sạch.

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
`annotations/0000f77c-6257be58.json`. Một frame tại timestamp10.000 có nhiều object.
`id` là ID object trong annotation, không phải class ID.
Class lấy từ `category` và map về integer như bảng tiếp theo.
`attributes` chứa trạng thái object; `poly2d` (nếu có) không dùng cho detection.

### Ánh xạ và phân bố bbox

Background ID0 chỉ dùng trong detector head; không phải một object được annotate.
Ánh xạ legacy: `person→pedestrian`, `motor→motorcycle`, `bike→bicycle`.
Head có11 logits = background +10 foreground classes.

Bảng đếm **bbox**, không phải ảnh. Một ảnh có thể nhiều bbox cùng/khác lớp.
Số đo sau dedup ảnh và quy tắc bbox giống notebook A2.

| Detector ID | Class | Category nguồn | Train bbox | Dev bbox | Test bbox |
|---:|---|---|---:|---:|---:|
| 1 | pedestrian | person | 91,434 | 13,265 | 24,650 |
| 2 | rider | rider | 4,522 | 649 | 1,294 |
| 3 | car | car | 714,099 | 102,522 | 205,150 |
| 4 | truck | truck | 30,012 | 4,239 | 8,704 |
| 5 | bus | bus | 11,688 | 1,597 | 3,217 |
| 6 | train | train | 136 | 15 | 28 |
| 7 | motorcycle | motor | 3,002 | 452 | 841 |
| 8 | bicycle | bike | 7,227 | 1,007 | 1,998 |
| 9 | traffic light | traffic light | 186,294 | 26,887 | 52,840 |
| 10 | traffic sign | traffic sign | 239,947 | 34,909 | 69,006 |

Lớp car chiếm phần lớn bbox; train/rider/motorcycle hiếm. Không cân bằng lại bằng
cách xóa các lớp phổ biến; dùng per-class AP để đọc chất lượng mỗi lớp.

### Số ảnh chứa từng lớp

Mỗi ảnh chỉ đếm một lần cho một class, dù class đó có nhiều bbox.
Tổng qua các class có thể lớn hơn số ảnh vì đây là detection multi-label.

| Class | Train ảnh | Dev ảnh | Test ảnh |
|---|---:|---:|---:|
| pedestrian | 22,103 | 3,220 | 6,213 |
| rider | 3,590 | 515 | 1,004 |
| car | 69,199 | 9,877 | 19,776 |
| truck | 18,915 | 2,687 | 5,500 |
| bus | 9,003 | 1,242 | 2,459 |
| train | 105 | 14 | 26 |
| motorcycle | 2,284 | 334 | 640 |
| bicycle | 4,350 | 578 | 1,182 |
| traffic light | 39,278 | 5,652 | 11,051 |
| traffic sign | 57,241 | 8,219 | 16,432 |

### Tọa độ và cache detection

Theo protocol local, chuyển source inclusive `[x1,y1,x2,y2]` thành bbox liên tục
`[x1,y1,x2+1,y2+1]`, clamp về kích thước ảnh, chuyển float32 và loại box không còn
diện tích. Có554 box source trùng endpoint trên ít nhất một trục;553 vẫn hợp lệ
sau chuyển đổi. Loại đúng1 box train nằm ngoài mép (x1=x2=1280), ảnh
`9f68b0a6-b2c427e6`, object ID4. JSON gốc giữ nguyên; cache metadata ghi lý do loại.
Occluded/truncated vẫn là positive; chỉ crowd/ignored tường minh mới ignore.

`samples.csv` có các cột:

```csv
name,image,annotation,width,height,frames,boxes,polygons,image_sha256,annotation_sha256
```

Đường dẫn tương đối với split; ``name`` là stem; boxes/polygons là số object source,
không phải số bbox sau lọc. SHA256 cho phép kiểm tra JPEG/JSON đồng bộ/không đổi.

A2 tự build hoặc dùng lại `detection_cache/{train,val,test}.npz` để tránh parse JSON
có polygon mỗi epoch. Đây là **dữ liệu dẫn xuất local**, không phải output model;
không copy ảnh hoặc JSON sang cache.

| Array NPZ | Kiểu/shape | Vai trò |
|---|---|---|
| `boxes` | float32 `(M,4)` | Bbox đã chuyển/clamp |
| `labels` | int64 `(M,)` | Detector class ID1–10 |
| `iscrowd` | int64 `(M,)` | Crowd flag |
| `offsets` | int64 `(N+1,)` | Bbox của ảnh i nằm trong `[offsets[i]:offsets[i+1]]` |

NPZ bám thứ tự raw samples.csv: train70.000/val10.000/test20.000;
dedup ảnh diễn ra trong danh sách loader, nên **M raw cache khác tổng bbox sạch**:
train1.288.404/val185.578/test367.728. JSON cạnh NPZ giữ annotation rules,
checksum arrays/source manifest và các trường hợp box bị loại. Không sửa/xóa cache
mà không sửa metadata tương ứng; có thể bỏ cả cặp NPZ/JSON để notebook xây lại.
Metric: COCO-style AP50:95, AP50/AP75, object-size AP, AR100, per-class AP/PR.
A2 resize short side640/max1280 trong model; ảnh JPEG trên đĩa vẫn1280×720.
Đây là kết quả **local legacy**, không phải official benchmark score.

### Kiểm tra nguồn local và tải lại

Đã kiểm tra CRC ZIP members, decode100.000 JPEG, parse100.000 JSON, name/pair,
SHA256 sau giải nén và manifests. verification.json ghi missing_images=0,
missing_annotations=0; issues.csv ghi4 ảnh trùng và bbox có endpoint trùng.
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

Các file dưới pretrained là weights **ImageNet** để khởi tạo backbone; project
không tải hoặc dùng ảnh ImageNet trực tiếp như một dataset train riêng.

| Model | Notebook | File/nguồn weights |
|---|---|---|
| DeiT-Small/16 non-distilled | A1 | [deit_small_patch16_224-cd65a155.pth](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| ResNet-50 (ImageNet-1K V1) | A1, A2 | [resnet50-0676ba61.pth](https://download.pytorch.org/models/resnet50-0676ba61.pth) |

Notebook dùng torch.hub cache tại datasets/pretrained; chỉ tải weights còn thiếu.
A1 dùng ResNet-50 bottleneck3/4/6/3 và DeiT-Small D384/H6/depth12, thay head43classes; A2 dùng backbone ResNet50 với detector head11classes.
Các weights này bị Git ignore, khác best checkpoint của lần train (chỉ trong RAM).

## Cache cũ và công bố

Folder VOC cũ (nếu có local) không được notebook nào trong project hiện tại sử dụng.
Giữ lại trên máy theo yêu cầu giữ dataset; không tính nó vào số liệu thực nghiệm và
không push. verification_summary/metadata phụ cũng giữ local và bị ignore.

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
