# CO5085 · Advanced Deep Learning

Project thực nghiệm gồm năm notebook độc lập và báo cáo tổng. Code viết trong từng
notebook; mỗi file chạy end to end bằng **Restart Kernel → Run All** với dữ liệu local.
Kết quả được lưu trong output cell của chính notebook khi bấm Save.

## Cấu trúc

```text
CO5085/
├── assignment/
│   ├── exercise-vne.pdf
│   ├── assignment1-vne.pdf
│   └── assignment2-vne.pdf
├── public/
│   └── index.html
├── source_code/
│   ├── E1.ipynb
│   ├── E2.ipynb
│   ├── E3.ipynb
│   ├── A1.ipynb
│   └── A2.ipynb
├── report/
│   ├── CO5085_Report.pdf
│   └── source_latex/
│       ├── main.tex
│       ├── preamble.tex
│       ├── references.bib
│       ├── chapters/
│       └── assets/
├── datasets/
│   ├── README.md                 # File duy nhất của datasets được đưa vào Git
│   └── ...                       # Dữ liệu/cache chỉ tồn tại local
├── README.md
├── .gitignore
└── .nojekyll                     # GitHub Pages phục vụ nguyên các file tĩnh
```

`source_code` chỉ chứa năm file `.ipynb`. Không dùng module/helper Python bên ngoài,
không đọc notebook khác, không cần chạy theo thứ tự E1 → E2 → E3 → A1 → A2.
Các notebook chỉ dùng thư viện đã cài và dataset tương ứng.

## Bài tập và dữ liệu

| Notebook | Thí nghiệm | Dataset | Train / validation / test | Số run |
|---|---|---|---|---:|
| [E1](source_code/E1.ipynb) | Softmax classifier, MLP, CNN-A, CNN-B | Fashion-MNIST | 54.000 / 6.000 / 10.000 | 4 |
| [E2](source_code/E2.ipynb) | MSA manual/PyTorch × 3 tokenizer; H=2/4/8 | Fashion-MNIST | 54.000 / 6.000 / 10.000 | 8 |
| [E3](source_code/E3.ipynb) | LSTM/GRU × 3 representation; MLP/CNN-B nội bộ | Fashion-MNIST | 54.000 / 6.000 / 10.000 | 8 |
| [A1](source_code/A1.ipynb) | ResNet-50/DeiT-Small, scratch/transfer/freeze, augmentation/seeds | GTSRB | 31.380 / 7.829 / 12.630 | 14 |
| [A2](source_code/A2.ipynb) | Faster R-CNN C5/FPN, anchors nhỏ, unfreeze | BDD100K legacy local | 69.998 / 9.998 / 20.000 | 4 |

Tổng **38 run**. Cấu hình mặc định dùng đầy đủ các split. Validation dùng để chọn
checkpoint/hyperparameter; test không tham gia lựa chọn. A2 loại ảnh trùng bằng
SHA256 trong manifest, giữ nguyên ảnh/JSON nguồn. Nhãn BDD là bản legacy người dùng
cung cấp; metric test local không được gọi là official Detection 2020 benchmark.

Xem [README dataset](datasets/README.md) để biết nguồn tải, định dạng file, phân bố
từng nhãn, ví dụ JSON/CSV, cách chia dữ liệu và các quy tắc kiểm tra.

## Chạy local

1. Mở riêng notebook cần chạy trong `source_code`.
2. Chọn interpreter **Python 3.14** trên máy, có PyTorch CUDA.
3. Giữ dữ liệu đã xử lý tại `datasets` theo README dataset. Notebook tự tìm project
   root từ đề PDF trong `assignment`, kể cả khi cwd là `source_code`.
4. **Restart Kernel → Run All**; đọc GPU preflight và tiến trình train/validation/test.
5. **Save notebook** sau khi chạy để lưu bảng/hình/log hiển thị vào `.ipynb`.

Thư viện: `torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`, `IPython`;
A2 cần thêm `pycocotools`. Dùng bản đang cài cho Python 3.14; chỉ cài package còn thiếu
bằng đúng interpreter (`python -m pip install <package>`). Không có môi trường
Python 3.13 riêng trong project. Windows dùng `NUM_WORKERS=0` cho Dataset khai báo
trong notebook. Không tự cài hoặc hạ phiên bản thư viện khi Run All.

E1–E3: CUDA bắt buộc, batch128, tối đa100 epoch, early stopping theo validation loss,
patience4, min_delta0,0001. Một khung text/model cập nhật tại chỗ; hiển thị
GPU/tensor/gradient, epoch/batch, loss/accuracy, updates, AMP skips, VRAM và ETA.
Bảng/đường cong có tên model/legend; confusion matrix có số ảnh và tỷ lệ theo hàng.

A1: ResNet-50 và DeiT-Small/16 non-distilled, ảnh224×224. Batch cố định128 cho
train/validation/test, accumulation1, effective batch128. Đo kiểm bên ngoài notebook
trên RTX A3000 12GB với AMP backward/AdamW states cho thấy scratch/full dùng khoảng
6.898MiB reserved (ResNet-50) và6.962MiB (DeiT-Small), tổng GPU khoảng72–73% khi đo.
Đánh giá FP32 batch128 đã được kiểm tra khi optimizer states còn resident.
Tiến trình hiển thị model, batch/epoch, CUDA, metric, VRAM và checkpoint.
AdamW, scratch40/pretrained25 epoch, patience4, min_delta0,0001, warmup/cosine và
clip1. Tám run chính so sánh CNN/Transformer, scratch/pretrained, head/partial/full;
hai no-augmentation và bốn run lặp tạo ba seed42/43/44 cho cấu hình chọn bằng val.
Test được gọi sau khi khóa14 run; báo metric full test và12.622 ảnh không trùng SHA256.
GPU inference benchmark giữ batch32 chung, độc lập batch đánh giá. Không tăng batch
vượt effective128 chỉ để đầy VRAM ở chế độ head-only; thời gian/throughput cũng cần đọc.
A2: micro-batch2 × accumulation2, SGD,12 epoch,640/max1280; nếu thiếu VRAM, chỉnh
micro-batch1/accumulation4 rồi Restart Kernel và chạy lại toàn bộ.
A2 early stopping theo validation AP50:95, patience4, min_delta0,0001; giữ checkpoint AP cao nhất.

## Kết quả, báo cáo và Git

Checkpoint tốt nhất là bản sao weights trong **CPU RAM**, được nạp cho validation/test
của cùng lần chạy. E2/E3 giữ đủ checkpoint để test/attention sau khi train các model.
Không ghi weights/history/CSV/PNG/log ra project; không có folder `outputs` hoặc
resume qua phiên kernel. Log chi tiết E1–E3/A1 giữ trong `EXECUTION_LOG` ở RAM.
Bấm Save mới lưu output cell vào notebook; Restart Kernel xóa các biến RAM.

[PDF báo cáo tổng](report/CO5085_Report.pdf) và [source LaTeX](report/source_latex/main.tex)
nằm trong `report`. `report/source_latex/assets` dành cho
hình/bảng của báo cáo, biên tập riêng sau khi có kết quả; notebook không tự ghi vào đây.
PDF hiện có chưa được biên dịch lại theo các lần Run All mới.

`.gitignore` loại toàn bộ dữ liệu/cache trong `datasets`, chỉ cho phép
`datasets/README.md`; đồng thời loại cache Python/Jupyter/Ruff và file build LaTeX.
Notebook với output cell, đề PDF, HTML, PDF báo cáo, source/assets LaTeX được phép
đưa vào Git. Không dùng `git add -f datasets`. Nếu repo remote đã track raw data trước
đó, cần bỏ chúng khỏi Git index trước khi push; ignore không tự bỏ file đã track.
Mọi thay đổi hiện tại ở local, chưa push Git.

## GitHub Pages

[Trang HTML](public/index.html) chứa CSS/JS inline và liên kết tương đối tới
`../source_code`, `../report`, `../assignment` và README dataset.

Để giữ đúng năm folder và không thêm workflow, sau khi duyệt/push hãy chọn
**Settings → Pages → Deploy from a branch → nhánh publish → /(root)**.
Truy cập trang tại **https://dangtai111325.github.io/CO5085/public/**.
Nếu cấu hình cũ upload `docs` bằng workflow, cần chuyển publishing source theo hướng
dẫn trên; không tiếp tục dùng đường dẫn `docs`.

GitHub chỉ hỗ trợ folder branch source là `/(root)` hoặc `/docs`, không chọn trực tiếp
`/public` trong menu này. [Tài liệu GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
`.nojekyll` ở root giúp phục vụ các tài nguyên nguyên bản. Cấu hình remote chưa được đổi.

Xem local bằng terminal tại CO5085:

```powershell
python -m http.server 5085 --bind 127.0.0.1
```

Mở `http://127.0.0.1:5085/public/`. Nếu port đã dùng, chọn port khác.

## Kiểm chứng

E1–E3 đã chạy end to end ngày **07/10/2026** bằng Python 3.14.7, PyTorch
2.11.0+cu128 và NVIDIA RTX A3000 12GB Laptop GPU. Mỗi model dùng đầy đủ
54.000 train / 6.000 validation / 10.000 test; không dùng subset thử nghiệm.

| Notebook | Model đã huấn luyện | Epoch thực tế | Optimizer updates thực tế |
|---|---:|---:|---:|
| E1 | 4 | 108 | 45.574 |
| E2 | 8 | 259 | 109.290 |
| E3 | 8 | 238 | 100.434 |
| **Tổng** | **20** | **605** | **255.298** |

Output cell đã lưu đủ tiến trình CUDA, bảng metric, learning curves, ma trận có
số ảnh/tỷ lệ, ảnh lỗi và phân tích đặc thù từng bài. Kiểm tra xác nhận mỗi epoch
đi qua đủ 54.000 ảnh/422 batch, metric hữu hạn, checkpoint có validation loss
thấp nhất và mỗi confusion matrix test chứa đúng 10.000 ảnh. Số optimizer updates
đã tính các batch bị AMP bỏ qua. Không có error/stderr trong output notebook;
bố cục các hình đã được kiểm tra trực quan. A1/A2 không được huấn luyện trong
lần kiểm chứng này.

A1 dùng ResNet-50 và DeiT-Small; phần kiểm tra GPU/VRAM và training loop nhỏ
được thực hiện riêng trong tệp tạm, không đưa metric thử vào báo cáo. Notebook
bàn giao có output trống và dùng đủ31.380/7.829/12.630 ảnh với ngân sách40/25 epoch
để người dùng tự Run All; chưa chạy huấn luyện full A1. Batch128 được cố định trong code cho train/val/test;
bảng/tiến trình ghi cấu hình và mức sử dụng tài nguyên thực tế.

### Đối chiếu yêu cầu E1–E3

| Đề bài | Code và output đã có |
|---|---|
| E1 §2.2 | Flatten softmax/MLP; CNN; loop tự viết; val/test; accuracy/capacity; loss/accuracy; confusion matrix và ảnh lỗi |
| E2 §3.2 | Manual MSA từ phép toán cơ bản; PyTorch reference; shallow classifier; Rows/Patch/CNN-stem; so sánh từng cặp cùng tokenizer và tokenizers theo accuracy/thời gian |
| E3 §4.2 | LSTM/GRU; Rows/Columns/Patch; MLP/CNN-B cùng kiến trúc/split với E1 nhưng train độc lập; bảng accuracy/params/time và thảo luận phụ thuộc theo chuỗi |
| Chung §1.2–1.4, §5 | Full Fashion-MNIST cùng split, seed/config/log, loop forward/loss/backward/step, README; báo cáo PDF/Pages còn cần cập nhật và publish sau khi duyệt |

Đây là đối chiếu phần thực nghiệm, không xác nhận hoàn tất điều kiện nộp bài.
Đề §1.6 giới hạn AI ở vai trò hỗ trợ và yêu cầu khai báo; công cụ AI đã tham gia
soạn code, kiểm tra và trình bày trong quá trình xây dựng project. Người nộp cần
khai báo đúng phạm vi, hiểu/bảo vệ sản phẩm và đối chiếu quy định môn học.

Các kiểm tra gồm cấu trúc, compile từng cell, schema notebook, Ruff và đường dẫn
liên kết local. Phân bố nhãn trong README dataset được đếm từ labels.npy/samples.csv/
detection cache local; không lấy số từ kết quả model. Người thực hiện cần kiểm
chứng, hiểu và diễn giải kết quả, đồng thời ghi nhận AI hỗ trợ theo yêu cầu đề bài.
