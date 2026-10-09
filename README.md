# CO5085 · Học sâu và ứng dụng trong thị giác máy tính

Thực nghiệm E1–E3, A1 và A2 gồm năm notebook độc lập và một báo cáo tổng.
Mỗi notebook chứa toàn bộ code cần thiết; bảng, đồ thị và ảnh kết quả được lưu
trong output cell sau khi chạy và Save.

## Nội dung

| Notebook | Phương pháp | Dataset | Train / validation / test | Lượt train |
|---|---|---|---|---:|
| [E1](source_code/E1.ipynb) | Softmax, MLP, CNN-A, CNN-B | Fashion-MNIST | 54.000 / 6.000 / 10.000 | 4 |
| [E2](source_code/E2.ipynb) | MSA thủ công/PyTorch; Rows, Patch, CNN-stem; H=2/4/8 | Fashion-MNIST | 54.000 / 6.000 / 10.000 | 8 |
| [E3](source_code/E3.ipynb) | LSTM/GRU theo hàng, cột, patch; baseline MLP/CNN-B | Fashion-MNIST | 54.000 / 6.000 / 10.000 | 8 |
| [A1](source_code/A1.ipynb) | ResNet-50 và DeiT-Small/16; scratch, transfer, freeze, augmentation, seeds | GTSRB, 43 lớp | 31.380 / 7.829 / 12.630 | 14 |
| [A2](source_code/A2.ipynb) | RetinaNet và Faster R-CNN, cùng ResNet-50/FPN | VOC2012, 20 lớp | 9.232 / 1.154 / 1.154 | 2 |

Tổng cộng **36 lượt train, 849 epoch và 334.186 optimizer updates**.
Validation chọn weights/cấu hình; test dùng để đánh giá sau lựa chọn.
A1 chia train/validation theo track biển báo và giữ test nguồn; báo thêm kết quả
trên 12.622 ảnh test không trùng SHA256 với train. A2 dùng toàn bộ 11.540 ảnh
VOC2012 trainval có nhãn, chia holdout nội bộ 80/10/10, seed 42; không phải
test chính thức hoặc điểm leaderboard VOC2012.

## Kết quả chính

| Bài | Kết quả test |
|---|---|
| E1 | Softmax 84,27%; MLP 89,45%; CNN-A 91,66%; CNN-B 91,97% accuracy |
| E2 | MSA PyTorch + CNN-stem H=4: 90,78% accuracy |
| E3 | LSTM rows 89,70%; GRU rows 90,06%; CNN-B baseline 91,97% accuracy |
| A1 | Pretrained full seed 42: ResNet-50 98,37%, DeiT-Small 98,92% accuracy; macro-F1 0,9743 / 0,9790 |
| A2 | RetinaNet / Faster R-CNN: VOC mAP50 **71,58% / 73,07%**; COCO-style AP50:95 **41,36% / 40,75%** |

A1 trung bình ba seed 42/43/44: ResNet-50 98,44±0,29%, DeiT-Small 98,95±0,33%
accuracy (độ lệch chuẩn mẫu). A2 latency median batch 1 là 47,35 / 48,92 ms;
F1 tại confidence 0,30 là 0,5857 / 0,5440. Các phép đo dùng RTX A3000 12GB Laptop.
Điều kiện đo, bảng từng lớp, đường cong và giới hạn so sánh có trong
[báo cáo PDF](report/CO5085_Report.pdf).

## Cấu trúc

```text
CO5085/
├── assignment/              # Đề PDF và báo cáo mẫu
├── public/index.html        # Trang giới thiệu, liên kết code và báo cáo
├── source_code/             # E1.ipynb, E2.ipynb, E3.ipynb, A1.ipynb, A2.ipynb
├── report/
│   ├── CO5085_Report.pdf
│   └── source_latex/        # main, preamble, architectures, chapters, bibliography, assets
├── datasets/
│   ├── README.md            # File duy nhất trong datasets được đưa vào Git
│   └── ...                  # Dữ liệu và pretrained weights trên máy
├── README.md
├── .gitignore
└── .nojekyll
```

`source_code` chỉ có năm notebook, không dùng module Python của project bên ngoài
và không đọc notebook khác. Không cần chạy theo thứ tự. Notebook không xuất file
weights/history/CSV/PNG; best weights giữ trong RAM cho cùng phiên chạy.
`report/source_latex/assets` chứa hình/bảng phục vụ biên tập báo cáo, được trích
từ output đã lưu. Các notebook không ghi vào thư mục này.

## Chạy notebook

1. Chuẩn bị dữ liệu theo [datasets/README.md](datasets/README.md).
2. Mở notebook và chọn **Python 3.14 có PyTorch CUDA**.
3. **Restart Kernel → Run All**; theo dõi khung tiến trình train/validation/test.
4. **Save** để giữ output trong `.ipynb`.

Môi trường thực nghiệm: Python 3.14.7, PyTorch 2.11.0+cu128, torchvision
0.26.0+cu128, NumPy 2.5.2, Matplotlib 3.11.2, Pillow 12.3.0; A2 dùng thêm
`pycocotools`. Chỉ cài thư viện còn thiếu bằng đúng interpreter
(`python -m pip install <package>`). Windows dùng `NUM_WORKERS=0`.
Notebook tự tìm project root từ đề PDF trong `assignment`.

Các tham số tạo ra số liệu báo cáo:

| Bài | Batch train / eval | Accumulation | Giới hạn epoch | Early stopping |
|---|---|---:|---|---|
| E1–E3 | 128 / 128 | 1 | 100 | Val loss, patience 10, min_delta 0,0001 |
| A1 | 128 / 128 | 1 | Scratch 40; pretrained 25 | Val loss, patience 4, min_delta 0,0001 |
| A2 | 4 / 4 | 3 | 30 | Val VOC mAP50, patience 2, min_delta 0,001 |

E1–E3 có mặc định patience 4 trong code; dùng patience 10 nếu tái lập đúng
thiết lập của các bảng đã lưu. Seed cố định không bảo đảm kết quả giống từng
bit trên GPU/thư viện khác.

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
ma trận có số và gallery GT/dự đoán/lỗi. COCO-style là cách tính trên VOC2012.

## Báo cáo và tài liệu

- [Báo cáo tổng E1–E3, A1, A2](report/CO5085_Report.pdf)
- [Nguồn LaTeX](report/source_latex/main.tex) và [sơ đồ TikZ đen–trắng](report/source_latex/architectures.tex)
- [Đề E1–E3](assignment/exercise-vne.pdf), [đề A1](assignment/assignment1-vne.pdf), [đề A2](assignment/assignment2-vne.pdf)
- [FPN, CVPR 2017](https://arxiv.org/abs/1612.03144), [Focal Loss, ICCV 2017](https://arxiv.org/abs/1708.02002)

Biên dịch nguồn bằng XeLaTeX → BibTeX → XeLaTeX hai lượt, tại
`report/source_latex`. Giữ file build trong thư mục tạm; PDF công bố là
`report/CO5085_Report.pdf`. Manifest trong assets ghi cell/output nguồn của
hình và bảng. Kết quả được đối chiếu với output, không chạy lại training khi
biên tập báo cáo.

## Trang public và Git

[Trang GitHub Pages](https://dangtai111325.github.io/CO5085/) tổng hợp kết quả,
liên kết xem/tải năm notebook, báo cáo PDF, đề bài và thông tin dataset.
HTML nguồn ở [public/index.html](public/index.html). Xem trên máy:

```powershell
python -m http.server 5085 --bind 127.0.0.1
```

Mở `http://127.0.0.1:5085/public/`. Các file phục vụ website và công cụ publish
đều nằm trong `public`. Sau khi commit và push `main`, publish bằng:

```powershell
& .\public\publish.ps1
```

Script lấy các file đã commit, tạo bản website trong thư mục tạm rồi push lên
nhánh `gh-pages`; không đổi notebook hoặc nhánh đang làm việc. Trong
Settings → Pages, Source là **Deploy from a branch**, nhánh **gh-pages**, folder
**/(root)**. GitHub tự deploy khi nhánh này được cập nhật, không cần thư mục
`.github` trong project. Chạy `& .\public\publish.ps1 -Preview` nếu chỉ muốn
dựng bản xem thử trong thư mục tạm, chưa push.

Trang gốc `/CO5085/` và `/CO5085/public/` dùng cùng HTML; script điều chỉnh
liên kết tương đối cho trang gốc. Bản public chỉ chứa trang, notebook, PDF và
README; không chứa dataset local hoặc file build LaTeX. Nguồn LaTeX được xem
trên GitHub. Lần push `main` tiếp theo cần chạy lệnh publish để cập nhật website.

Git chỉ chứa đề bài, notebook/output, HTML, README, PDF và nguồn/assets LaTeX.
Toàn bộ raw data, pretrained cache và file build bị ignore; trong `datasets`
chỉ README được track. Không dùng `git add -f` với dataset.
Video demo A2 là sản phẩm nộp riêng theo đề; trang public ghi trạng thái video.

## Ghi nhận hỗ trợ

Công cụ AI hỗ trợ lập trình, kiểm tra code, tổng hợp output và biên tập báo cáo.
Số liệu lấy từ các notebook đã chạy và lưu. Người thực hiện chịu trách nhiệm
hiểu, kiểm chứng và bảo vệ sản phẩm, đồng thời ghi nhận phạm vi hỗ trợ theo
mục 1.6 đề E1–E3.
