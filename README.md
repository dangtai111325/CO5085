# CO5085 - Học sâu và ứng dụng trong thị giác máy tính

Repository coursework cho học kỳ 261 (2026-2027), tổ chức bám theo ba đề E1-E3, A1 và A2. Mục tiêu của khung này là: **clone -> kiểm tra môi trường -> chạy smoke test -> chạy notebook/thí nghiệm thật -> commit artifact -> cập nhật báo cáo**.

## 1. Trạng thái lựa chọn hiện tại

| Hạng mục | Lựa chọn |
|---|---|
| E1-E3 dataset | Fashion-MNIST |
| A1 dataset | **Đề xuất hiện tại:** GTSRB (traffic-sign recognition) |
| A1 models | Custom ResNet-18 vs custom ViT/DeiT-Tiny; scratch/pretrained/freeze/partial/full |
| A2 topic | Object Detection |
| A2 dataset | **Đề xuất hiện tại:** BDD100K detection |
| A2 paper | *Feature Pyramid Networks for Object Detection* (CVPR 2017); Faster R-CNN là framework nền |
| GPU mục tiêu | NVIDIA Precision A3000 12GB |

> Dataset A1/A2 vẫn được đánh dấu là đề xuất trong báo cáo cho đến khi nhóm xác nhận trước các mốc A1.1/A2.1. Không có kết quả hoặc thống kê nào được tự tạo.

## 2. Quick start

```bash
git clone https://github.com/dangtai111325/CO5085.git
cd CO5085
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Cài dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra GPU/môi trường:

```bash
python scripts/check_env.py
```

Chạy smoke test end-to-end bằng synthetic tensors (không tải dataset, không dùng kết quả này trong báo cáo):

```bash
python scripts/smoke_test.py
```

Build báo cáo:

```bash
bash scripts/build_report.sh
```

Trên Windows nếu chưa có `bash`, chạy trong Git Bash/WSL hoặc vào `report/source_latex` và chạy:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

sau đó copy `main.pdf` thành `report/CO5085_report.pdf`.

## 3. Cấu trúc repository

```text
CO5085/
├── common/                  # seed, training loop, metrics, plotting, checkpoint
├── E1/
│   ├── E1.ipynb
│   ├── src/
│   └── outputs/
├── E2/
│   ├── E2.ipynb
│   ├── src/
│   └── outputs/
├── E3/
│   ├── E3.ipynb
│   ├── src/
│   └── outputs/
├── A1/
│   ├── notebooks/           # 01 EDA -> 07 final figures
│   ├── src/
│   ├── configs/
│   └── outputs/
├── A2/
│   ├── notebooks/           # 01 prepare -> 10 paper reproduction
│   ├── src/
│   ├── configs/
│   └── outputs/
├── data/README.md           # dataset layout; dataset files are gitignored
├── assets/                  # toàn bộ hình dùng trong báo cáo
├── report/
│   ├── source_latex/        # toàn bộ source LaTeX
│   ├── results/             # CSV/JSON/table artifacts chọn để báo cáo
│   └── CO5085_report.pdf    # PDF build hiện tại
├── docs/                    # GitHub Pages
├── scripts/                 # env check, smoke test, report build, validation
└── tests/
```

## 4. Quy tắc code

- Không dùng `trainer.fit()` hoặc wrapper huấn luyện end-to-end.
- Training/validation/test loop được viết rõ bằng PyTorch primitives.
- E2 manual MSA tự viết Q/K/V, scaled dot-product, softmax và concat heads; `nn.MultiheadAttention` chỉ là reference.
- E3 có manual LSTM/GRU cell ngoài `nn.LSTM`/`nn.GRU` reference.
- A1 custom ResNet-18 và custom ViT-Tiny; pretrained chỉ là weight initialization, không thay thế training protocol.
- A2 không gọi `torchvision.models.detection.FasterRCNN` hoặc `fasterrcnn_resnet50_fpn(...)`; code tự ghép transform + backbone stages + custom FPN + custom anchors/RPN + RoI head/Fast R-CNN predictor. Chỉ các operator tối ưu cấp thấp như NMS/RoIAlign được dùng từ torchvision.
- Synthetic smoke data chỉ để test code; **không bao giờ đưa số liệu smoke vào báo cáo**.

## 5. Notebook workflow

Mọi notebook tuân theo chuỗi: **Environment -> Data/EDA -> Split/DataLoader -> Model sanity -> Train -> Validate/Test -> Visualize -> Export artifacts -> Report checklist**. A1/A2 được tách nhiều notebook để không phải re-run training chỉ để vẽ hình.

## 6. Báo cáo

Source: `report/source_latex/`  
Figures: `assets/`  
Final PDF: `report/CO5085_report.pdf`

Style được dựng theo report mẫu đã cung cấp: cover HCMUT, Times-like serif typography, header line, mục lục/list of figures/list of tables, caption kiểu `Bảng/Hình`, bảng `booktabs`, và placeholder màu đỏ cho dữ liệu/hình chưa có.

Theo đề E1-E3, ba bài dùng một PDF chung. Đề A1 và A2 mỗi đề đều nói nộp `code + báo cáo PDF`; do đó source được tập trung để có thể xuất master report hoặc tách riêng về sau nếu giảng viên yêu cầu.

## 7. GitHub Pages

Static site nằm tại `docs/index.html`. Workflow `.github/workflows/pages.yml` deploy `docs/` và copy `report/CO5085_report.pdf` vào site artifact. Pages phải được bật một lần trong **Settings -> Pages -> Source: GitHub Actions**.
