# CO5085 · Học sâu và ứng dụng trong thị giác máy tính

Thực nghiệm E1–E3, A1 và A2 gồm năm bài thực nghiệm độc lập và một báo cáo tổng.
Mỗi notebook chứa toàn bộ code cần thiết; bảng, đồ thị và ảnh kết quả được lưu
trong output cell sau khi chạy và Save.

## Nhóm thực hiện

| Thành viên | Mã số học viên |
|---|---|
| Đặng Nguyễn Tấn Tài | 2570357 |
| Lương Minh Duy | 2570083 |

**Giảng viên:** Lê Thành Sách. **Lớp:** CH01. **Học kỳ:** 261 (2026–2027).

## Nội dung

| Notebook | Phương pháp | Dataset | Train / validation / test | Lượt train |
|---|---|---|---|---:|
| [E1](source_code/E1.ipynb) | Softmax, MLP, CNN-A, CNN-B | Fashion-MNIST | 54.000 / 6.000 / 10.000 | 4 |
| [E2](source_code/E2.ipynb) | MSA thủ công/PyTorch; Rows, Patch, CNN-stem; H=2/4/8 | Fashion-MNIST | 54.000 / 6.000 / 10.000 | 8 |
| [E3](source_code/E3.ipynb) | LSTM/GRU theo hàng, cột, patch; baseline MLP/CNN-B | Fashion-MNIST | 54.000 / 6.000 / 10.000 | 8 |
| [A1](source_code/A1.ipynb) | ResNet-50 và DeiT-Small/16; scratch, transfer, freeze, augmentation, seeds | GTSRB, 43 lớp | 31.380 / 7.829 / 12.630 | 14 |
| [A2](source_code/A2.ipynb) | RetinaNet và Faster R-CNN, cùng ResNet-50/FPN | VOC2012, 20 lớp | 9.232 / 1.154 / 1.154 | 2 |

Số liệu trong báo cáo gồm **36 lượt train, 541 epoch và 210.351 optimizer updates**.
Validation chọn weights/cấu hình; test dùng để đánh giá sau lựa chọn.
A1 chia train/validation theo track biển báo và giữ test nguồn; báo thêm kết quả
trên 12.622 ảnh test không trùng SHA256 với train. A2 dùng toàn bộ 11.540 ảnh
VOC2012 trainval có nhãn, chia holdout nội bộ 80/10/10, seed 42; không phải
test chính thức hoặc điểm leaderboard VOC2012.

## Kết quả chính

| Bài | Kết quả test |
|---|---|
| E1 | Softmax 84,11%; MLP 89,45%; CNN-A 91,66%; CNN-B 91,45% accuracy |
| E2 | MSA thủ công + CNN-stem H=4: 90,50% accuracy |
| E3 | LSTM rows 89,70%; GRU rows 90,06%; CNN-B baseline 91,45% accuracy |
| A1 | Pretrained full seed 42: ResNet-50 97,83%, DeiT-Small 98,92% accuracy; macro-F1 0,9678 / 0,9790 |
| A2 | RetinaNet / Faster R-CNN: VOC mAP50 **70,88% / 73,16%**; COCO-style AP50:95 **39,13% / 40,98%** |

A1 trung bình ba seed 42/43/44: ResNet-50 97,95±0,21%, DeiT-Small 98,50±0,36%
accuracy (độ lệch chuẩn mẫu). A2 trên Tesla T4 có latency median batch 1
39,87 / 44,09 ms, tương ứng 25,08 / 22,68 FPS; micro-F1 tại confidence 0,30
là 0,6133 / 0,5295. E1–E3/A1 chạy trên RTX A3000 12GB Laptop.
Điều kiện đo, bảng từng lớp, đường cong và giới hạn so sánh có trong
[báo cáo PDF](report/CO5085_Report.pdf).

## Cấu trúc

```text
CO5085/
├── assignment/              # Đề PDF và báo cáo mẫu
├── public/                  # index.html và .nojekyll
├── source_code/             # Năm notebook E1–E3, A1 và A2
├── report/
│   ├── CO5085_Report.pdf
│   └── source_latex/        # main, preamble, architectures, chapters, bibliography, assets
├── datasets/
│   ├── README.md            # File duy nhất trong datasets được đưa vào Git
│   └── ...                  # Dữ liệu và pretrained weights trên máy
├── model/                   # Latest checkpoint từng thí nghiệm, chỉ local
├── README.md
└── .gitignore
```

`source_code` chứa năm notebook E1–E3, A1 và [A2](source_code/A2.ipynb).
Số liệu A2 trong báo cáo lấy từ output chạy trên Kaggle.
Mỗi file tự chứa code và không đọc notebook khác. A2 tự tạo worker
để train hai model song song, không cần upload module của project. Không cần chạy theo thứ tự. Notebook hiển thị bảng và hình trong output cell. Mỗi thí nghiệm giữ một file
`model/<bài>/<thí nghiệm>/latest.pt`, chỉ chứa trọng số tốt nhất theo validation.
File chỉ được thay khi validation cải thiện. `report/source_latex/assets` chứa hình/bảng phục vụ biên tập báo cáo, được trích
từ output đã lưu. A2 Kaggle xuất thêm PNG/CSV vào `/kaggle/working/report/A2`; checkpoint,
history và run config ở `/kaggle/working/model/A2`. Các notebook không ghi
vào thư mục assets LaTeX của project.

## Chạy E1–E3 và A1

1. Chuẩn bị dữ liệu theo [datasets/README.md](datasets/README.md).
2. Mở notebook và chọn **Python 3.14 có PyTorch CUDA**.
3. **Restart Kernel → Run All**; theo dõi khung tiến trình train/validation/test.
4. **Save** để giữ output trong `.ipynb`.

Môi trường thực nghiệm E1–E3/A1: Python 3.14.7, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, NumPy 2.5.2, Matplotlib 3.11.2, Pillow 12.3.0.
A2 cần thêm `pycocotools`. Chỉ cài thư viện còn thiếu bằng đúng interpreter
(`python -m pip install <package>`). Windows dùng `NUM_WORKERS=0`.

| Nhóm | Thư viện trực tiếp |
|---|---|
| Huấn luyện và ảnh | `torch`, `torchvision`, `numpy`, `Pillow` |
| Output notebook | `matplotlib`, `IPython` |
| COCO-style evaluator A2 | `pycocotools` |

Kiểm tra interpreter/kernel và GPU trước khi Run All:

```python
import sys, torch
print(sys.executable, sys.version)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Các phiên bản trên mô tả môi trường thực nghiệm; project dùng Python 3.14
global và thư viện đang cài, không yêu cầu tạo môi trường Python riêng.
Notebook local tự tìm project root từ đề PDF trong `assignment`.

Cấu hình tạo ra số liệu báo cáo:

| Bài | Batch train / eval | Accumulation | Giới hạn epoch | Early stopping |
|---|---|---:|---|---|
| E1–E3 | 128 / 128 | 1 | 100 | Val loss, patience 4, min_delta 0,0001 |
| A1 | 128 / 128 | 1 | Scratch 40; pretrained 25 | Val loss, patience 2, min_delta 0,001 |
| A2 Kaggle | 8 / 8 | 3 | 30 | Val VOC mAP50, patience 2, min_delta 0,001 |

A2 có effective batch 24; cấu hình trong bảng lấy từ cell code và
output train của A2.ipynb trên Kaggle.

Khi Run All, notebook nạp `latest.pt` nếu có rồi tiếp tục train từ trọng số
đó; nếu chưa có thì khởi tạo model theo cấu hình. Vòng epoch, optimizer,
learning-rate schedule và early-stopping counter bắt đầu cho lượt train hiện
tại. Không lưu số epoch, optimizer hoặc lịch sử vào file model. Validation
của trọng số nạp vào làm mốc để tránh ghi đè bằng model kém hơn. Best epoch 0
trong output nghĩa là trọng số đầu lượt train vẫn tốt nhất; train accuracy
của mốc này được đo trong eval mode. Các epoch 1 trở đi đo train accuracy
trong train mode như bình thường.

Checkpoint được ghi vào file tạm rồi thay thế nguyên tử. Để tái lập một thí
nghiệm từ khởi tạo, dùng `CHECKPOINT_ROOT` trống. Mỗi run có đường dẫn riêng,
kể cả các seed và mức freeze, nên không dùng lẫn trọng số. Seed cố định không bảo đảm kết quả giống từng bit trên GPU/thư viện khác.

A1 dùng ảnh 224×224, ResNet-50 và DeiT-Small/16 non-distilled. Hai họ đều có
scratch, pretrained head-only, partial và full; thêm no-augmentation và repeat
seeds cho full. A2 resize giữ tỷ lệ cạnh ngắn 600/max 1000, flip ngang train,
ImageNet normalization; SGD LR 0,01, momentum 0,9, weight decay 0,0001,
warmup một epoch/cosine, clip norm 10. Best weights được chọn trước test.

A2 RetinaNet dùng FPN P3–P7, 9 anchors/vị trí, Focal Loss + Smooth L1;
Faster R-CNN dùng P2–P6, RPN 3 anchors/vị trí, RoI Pool 7×7 và hai FC 1024.
Code giữ đúng lateral/top-down/smoothing và cách tạo P6/P7 của từng detector.
Nhãn difficult được học khi train, bỏ qua khi đánh giá. Bảng test có VOC AP,
12 COCO-style AP/AR, P/R/F1, AP từng lớp, FPS/latency/VRAM; kèm PR curves,
ma trận phần trăm theo hàng và gallery GT/dự đoán/lỗi. COCO-style là cách tính trên VOC2012.

## Chạy A2 trên Kaggle

1. Mở [A2.ipynb](source_code/A2.ipynb) với kernel Kaggle,
   chọn **GPU T4 x2** và bật Internet để tải weights khi thiếu.
2. Gắn dataset **A2 VOC2012** tại
   `/kaggle/input/datasets/dangtai1720/a2-voc2012`, gồm thư mục `voc2012`
   với ba CSV split và các JPEG/XML theo đường dẫn tương đối trong CSV.
3. Nếu mở file bằng VS Code, chọn **Existing Jupyter Server** với URL tương
   thích VS Code của session Kaggle, rồi chọn Python kernel. Không lưu URL/token vào Git.
4. **Run All**: RetinaNet train trên GPU 0, Faster R-CNN trên GPU 1; test và
   benchmark chạy lần lượt trên GPU 0 sau khi hai worker hoàn tất.
5. Save notebook có output và tải `/kaggle/working/report/A2` cùng checkpoint,
   history/config trong `/kaggle/working/model/A2` trước khi kết thúc session.

Metadata ghi Python 3.13.15; output không ghi phiên bản PyTorch/torchvision.
Ghi lại phiên bản thư viện khi tái lập, cùng cấu hình accelerator. Có 21/17
epoch, best 19/15; tổng 14.630 cập nhật. Hai worker train/validation đồng thời
trong khoảng 465,1 phút, không cộng thời gian riêng của chúng làm thời gian chờ.

## Báo cáo và tài liệu

- [Báo cáo tổng E1–E3, A1, A2](report/CO5085_Report.pdf)
- [Nguồn LaTeX](report/source_latex/main.tex) và [sơ đồ TikZ đen–trắng](report/source_latex/architectures.tex)
- [Đề E1–E3](assignment/exercise-vne.pdf), [đề A1](assignment/assignment1-vne.pdf), [đề A2](assignment/assignment2-vne.pdf)
- [FPN, CVPR 2017](https://arxiv.org/abs/1612.03144), [Focal Loss, ICCV 2017](https://arxiv.org/abs/1708.02002)

Biên dịch nguồn bằng XeLaTeX → BibTeX → XeLaTeX hai lượt, tại
`report/source_latex`. Giữ file build trong thư mục tạm; PDF công bố là
`report/CO5085_Report.pdf`. Manifest trong assets ghi cell/output nguồn của
hình và bảng. Các hình là ảnh độc lập từ output hoặc được vẽ trực
tiếp từ số liệu/dữ liệu gốc, không cắt từ ảnh ghép. Bảng và hình được đối
chiếu với output của phiên Run All trên toàn bộ dữ liệu.

## Trang public và Git

[Trang GitHub Pages](https://dangtai111325.github.io/CO5085/) tổng hợp kết quả,
protocol, đối chiếu yêu cầu đề, notebook, PDF, nguồn dữ liệu và paper.
Mỗi mục E1–E3/A1/A2 có liên kết trực tiếp tới phần tương ứng trong báo cáo.
Bảng đầy đủ có thể cuộn ngang; các hình mở được ở kích thước gốc.
Video demo A2 chưa có liên kết công bố; giấy phép cụ thể và quyền phân phối
lại archive GTSRB chưa được xác minh (xem nguồn trong README dataset).

HTML nguồn ở [public/index.html](public/index.html). Xem trên máy:

```powershell
python -m http.server 5085 --bind 127.0.0.1
```

Mở `http://127.0.0.1:5085/public/`. Giữ cấu trúc tương đối giữa `public`,
`assignment`, `source_code` và `report` để các liên kết tài liệu hoạt động.
Dataset và pretrained weights không được đóng gói vào website.

Workflow [Pages và release](.github/workflows/pages.yml) chạy khi push `main`:
đóng gói HTML/tài liệu/ảnh, kiểm tra đường dẫn và anchor, deploy Pages,
kiểm tra trang/PDF công khai rồi cập nhật GitHub Release. Trang chính được
phục vụ trực tiếp tại `/CO5085/`; `/CO5085/public/` vẫn truy cập được.
[Release](https://github.com/dangtai111325/CO5085/releases/latest) cung cấp bản
phát hành và PDF đính kèm. Workflow dùng tag `v2026.10.17`; khi release đã
tồn tại, workflow cập nhật ghi chú và trạng thái latest, giữ tag và asset
của lần tạo. Mã nguồn hiện hành nằm trên `main`.

Git chỉ chứa đề bài, notebook/output, HTML, README, PDF và nguồn/assets LaTeX.
Toàn bộ raw data, pretrained cache, thư mục `model` và file build bị ignore; trong `datasets`
chỉ README được track. Không dùng `git add -f` với dataset.
Video demo A2 là sản phẩm nộp riêng theo đề; trang public ghi trạng thái video.

## Ghi nhận hỗ trợ của công cụ AI

Phần ghi nhận này áp dụng chung cho **E1, E2, E3, A1 và A2**, cùng báo cáo
tổng và tài liệu của project. Công cụ AI hỗ trợ các công việc sau:

- Tổ chức, format code và Markdown trong notebook; hỗ trợ viết mã PyTorch
  theo phương pháp, kiến trúc và cấu hình được nhóm thực hiện chỉ định;
  triển khai vòng lặp train, validation và test trong từng notebook độc lập.
- Triển khai các tiện ích theo yêu cầu: đặt seed, theo dõi tiến trình và ETA,
  ghi nhận metric, lưu best weights vào `latest.pt` và nạp weights để train tiếp.
- Kiểm tra cú pháp, chạy thử pipeline trên dữ liệu nhỏ và xử lý lỗi lập trình;
  đối chiếu phép toán với bản tham chiếu ở các phần cần kiểm chứng.
- Tổng hợp output thành bảng so sánh accuracy, F1, số tham số, thời gian và
  các chỉ số đánh giá object detection; giữ rõ điều kiện đo và đơn vị.
- Viết mã vẽ learning curves, ma trận nhầm lẫn có số, ảnh mẫu, ảnh dự đoán
  và trường hợp sai; bố trí legend, nhãn trục, màu và nhóm hình để dễ đọc.
- Dựng sơ đồ TikZ từ định nghĩa mạng và kích thước tensor trong code;
  biên tập báo cáo theo mẫu và yêu cầu trình bày, gồm bố cục, caption,
  font, kích thước hình, nguồn LaTeX và diễn đạt phân tích từ số liệu thực nghiệm.
- Soạn hướng dẫn chuẩn bị dữ liệu, môi trường, chạy thí nghiệm và tra cứu
  kết quả; tổ chức thông tin trong README và trang giới thiệu của project.

Số liệu và hình thực nghiệm lấy từ output notebook. Nhóm thực hiện chịu
trách nhiệm xác định hướng nghiên cứu, kiểm chứng kết quả, hiểu và bảo vệ
mã nguồn cùng phần phân tích. Phạm vi hỗ trợ được ghi nhận theo mục 1.6
của [đề E1–E3](assignment/exercise-vne.pdf).
