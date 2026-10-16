# Biên tập và kiểm chứng báo cáo

Tài liệu bàn giao biên tập, không được chèn vào PDF học thuật.

## Phạm vi và dung lượng

- PDF trước: 160 trang. PDF bàn giao: 99 trang, giảm 61 trang (38.125%).
- Bố cục: 4 trang bìa/tóm tắt/mục lục; 52 trang Chương 1–7; 2 trang tài liệu tham khảo; 41 trang phụ lục.
- Giữ A4 dọc, body 12 pt, Times New Roman; công thức Cambria Math; thụt đầu dòng 1 cm và lề hiện có.
- Giữ một báo cáo tổng; thông tin nhóm được đồng bộ giữa bìa, metadata PDF, README và public HTML. Không chạy train/inference, chỉnh notebook/output, dataset hoặc checkpoint.
- Giảng viên: Lê Thành Sách. Nhóm thực hiện: Đặng Nguyễn Tấn Tài (2570357) và Lương Minh Duy (2570083), lớp CH01.
- `A2.kaggle.ipynb` là nguồn kết quả A2; cấu hình thực chạy batch/eval 8, accumulation 3, effective batch 24, patience 2, min_delta 0.001.
- Bản TikZ hiện có được giữ làm cơ sở; hình khối/kênh/cổng/head vẫn theo implementation. Hình dùng mực đen và nền trắng.

## Changelog theo file và nội dung

| File / mục | Giữ | Gộp / bỏ / sửa câu |
|---|---|---|
| `main.tex`, phần đầu | Bìa, abstract, TOC | Bỏ danh sách hình/bảng. Chuẩn hóa decimal/thousands; phiên bản và năm không đổi. |
| Chương 1 | Mapping đề, split, mã kiểm chứng trong metadata, normalization, môi trường, CE/F1, audit updates, checkpoint | Bỏ bar chart phân bố Fashion-MNIST cân bằng. Viết gọn protocol; giải thích best-weights khác full resume một lần. |
| `chapters/e1.tex` | Bốn mạng, loop tự viết, metric/cost, loss/accuracy, cặp nhầm, lỗi và giới hạn | Hai figure kiến trúc (Softmax/MLP; CNN-A/CNN-B). Bỏ bar charts quality/capacity. Ghép curves và hai gallery. Số/tần suất lỗi đọc từ bảng, không lặp nguyên bảng trong văn bản. |
| `chapters/e2.tex` | Ba tokenizer, Q/K/V, encoder, parity, tám cấu hình, bảng paired comparison/head ablation, attention/entropy/lỗi | Bỏ subsection đơn lẻ và hai paired bar charts. Ghép head loss/accuracy, CNN-stem curves, attention và lỗi. Hai sample nhỏ đi cùng attention. Rút đoạn diễn giải bảng; giữ timing, N² và giới hạn một seed/capacity. |
| `chapters/e3.tex` | Rows/Columns/Patch, classifier, công thức/cổng LSTM/GRU, parity, baseline, metric/cost/lỗi | Một figure ba cách quét; một figure hai cell. Bỏ subsection đơn lẻ, bảng paired/baseline dẫn xuất. Giữ delta GRU–LSTM trong câu và hai baseline ở bảng chính. Ghép curves/lỗi; không suy long-range dependency từ accuracy. |
| `chapters/a1.tex` | GTSRB/track split/SHA256, augmentation, hai backbone, regime/LR/freeze, metric/cost, ablation/seeds, lỗi và duplicate sensitivity | Gallery độc lập 0–21 và 22–42 giữ đủ 43 lớp. Bỏ scatter sizes, capacity/augmentation/seed bar charts và hai bảng delta dẫn xuất. Ghép curves/matrices/errors; rút văn bản theo bằng chứng–trade-off–giới hạn. |
| `chapters/a2.tex` | VOC/XML/difficult/truncated, resize/bbox, hai paper/ablation, anchors/matching/heads, train/selection, AP/AR/F1, benchmark và failure cases | Class counts thành grouped bars từ output; ghép bbox histograms và preprocessing. Bỏ image-size/AP/VRAM bar charts trùng bảng. Ghép hai FPN và hai head; ghép validation/component-loss/per-class curves. Bỏ hai subsection đơn lẻ, gọn benchmark nhưng giữ phạm vi đo. |
| Chương 7 | Tổng hợp trade-off, giới hạn và ghi nhận AI/trách nhiệm tác giả | Bốn đoạn kết luận và đoạn AI không đánh số; không đọc lại các accuracy. |
| `chapters/appendix_results.tex` | Toàn bộ audit/per-class/sensitivity tables, 20 Fashion-MNIST matrices, 20 PR curves, hai ma trận A2, 18 ảnh GT/prediction, reproducibility | Curves và matrices ghép cặp; head loss chuyển về E2. PR hai plot/trang. Sáu triplets theo đúng ID. Bỏ các lời dẫn trùng caption/protocol; tiêu đề gắn với nhóm hình. |
| `preamble.tex` | Font, body size, margins, portrait, indent | Macros subfigure/cặp/triplet, label/reference, float theo loại nội dung và khoảng cách caption tiêu chuẩn. Không stretch ảnh, trim/clip hay vspace âm hàng loạt. |
| `architectures.tex` | Tensor dimensions, gates, projections, FPN levels, RPN/RoI Pool/FC outputs | Chuẩn hóa số hiển thị; giữ tọa độ hình học. Bỏ badge E1 lặp caption, bỏ số hình hard-coded trong comment; thay grayscale fill bằng white. |
| `assets/*/*.tex` | Toàn bộ giá trị và precision các bảng | Chỉ chuẩn hóa định dạng số, không đổi class/model/order/units. Các bảng không được chèn vào PDF vẫn giữ trong assets. |
| `assets/report_figures.json`, `render_report_figures.py` | Provenance và khả năng dựng hình tĩnh | 14 hình độc lập từ output/CSV/pixel gốc: EDA, preprocessing, gallery và scan. Các PNG gốc đều giữ nguyên. Renderer chỉ đọc nguồn, không train/inference hay sửa notebook. |

## Đối chiếu yêu cầu học thuật

| Bài | Yêu cầu / bằng chứng cần chấm | Mục PDF |
|---|---|---|
| E1 | Softmax/MLP/CNN; loop tự viết; accuracy/F1, capacity, loss/accuracy, confusion và lỗi | 2.1–2.4; audit A.1 và matrices A.4 |
| E2 | MSA manual/PyTorch; Q/K/V; ba tokenizer; hai phép so sánh; head ablation; forward/gradient parity; accuracy và thời gian | 3.1–3.5; curves A.3, audit A.1, matrices A.4 |
| E3 | LSTM/GRU formulas/cells; Rows/Columns/Patch; MLP/CNN baseline; accuracy/parameters/cost; giới hạn phụ thuộc xa | 4.1–4.5; curves A.3, matrices A.4 |
| A1 | Dataset lớn/khác Fashion-MNIST; track split; CNN/Transformer, scratch/pretrained, ba freeze modes; accuracy/F1/cost; augmentation/seeds; duplicates/lỗi | 5.1–5.8; A.1–A.2 |
| A2 | Detection; >=2 phương pháp; VOC holdout/EDA/preprocessing; FPN/Focal Loss method/experiments/limitations; mAP50/AP50:95/AP75/scale/AR/PR/F1/FPS; failure cases | 6.1–6.10; class tables/matrices/PR/gallery A.5–A.9 |
| Chung | Protocol, nguồn dữ liệu/paper, test độc lập selection, checkpoint và tái lập, ghi nhận AI | 1.1–1.3; Chương 7; bibliography; A.10 |
| Sản phẩm nộp ngoài PDF | Code/landing page và video demo | Code/Pages tham chiếu như sản phẩm riêng; video chưa có, không khai đã nộp |

## Kiểm chứng và lựa chọn vì khả năng đọc

- Mọi số trong 1,764 token số của các bảng `.tex` được đối chiếu trước/sau cả giá trị lẫn số chữ số thập phân.
- Notebook/output và tài nguyên gốc được kiểm tra bằng SHA256; dataset/checkpoint bằng size/mtime, không ghi lại dữ liệu.
- Counts VOC cả 60 cặp split/lớp khớp output Kaggle; gallery GTSRB dùng đúng mẫu đầu của từng class trong manifest, kiểm chứng hash ảnh.
- Khoảng `[0,1]`, mức RoI `[2,5]`, danh sách scales, phiên bản, URL, ID và tọa độ TikZ không bị hiểu nhầm thành decimal/grouping.
- Không còn tiêu đề con đánh số đơn lẻ; chỉ giữ cặp 6.4.1/6.4.2 có nội dung thực.
- XeLaTeX/BibTeX chạy đủ lượt: không undefined citation/reference, duplicate label, Overfull, float quá cao hoặc missing Unicode glyph.
- Tất cả trang portrait; font PDF chỉ Times New Roman và Cambria Math; không có trang trắng.
- Đã render và kiểm tra toàn bộ 99 trang, đồng thời kiểm tra ở độ phân giải cao các sơ đồ, ma trận, gallery lớp và ảnh bbox. PDF có 70 nhóm hình, 45 bảng và 12 mục tài liệu tham khảo; số thứ tự và liên kết được biên dịch lại.
- Kết quả kiểm chứng máy đọc và SHA256 của PDF nằm trong `assets/report_audit.json`; giá trị kiểm tra trước/sau và các trường hợp loại trừ nằm trong `assets/number_format_audit.json`.
- PNG gốc không sửa/cắt. Hình cần đổi format số được dựng độc lập từ nguồn đã xác nhận, lưu provenance và mã dựng hình.
- Giữ PR hai plot/trang và hai ma trận A2 riêng trang, vì giảm thêm làm chữ/legend quá nhỏ. Phần phụ lục chiếm 41 trang để bảo toàn các kết quả người dùng yêu cầu giữ.
- Không có ngoại lệ decimal/thousands chưa xử lý trong các hình được sử dụng: các hình native còn lại đã có dấu chấm cho số lẻ; các EDA/preprocessing cần sửa được render lại từ nguồn. Các PNG không dùng vẫn giữ byte gốc.
- Một số Underfull hbox của URL/bibliography hoặc chuỗi thuật ngữ còn được LaTeX báo, nhưng không tràn/cắt chữ; đã kiểm tra trực quan.

## Dựng lại

Biên dịch `main.tex` bằng XeLaTeX → BibTeX → XeLaTeX đủ lượt. Nếu dùng thư mục build riêng, đặt thư mục build trước source trong TEXINPUTS để tránh đọc `.aux/.toc` của một build khác. PDF bàn giao chính là `report/CO5085_Report.pdf`.

Dựng hình tĩnh khi prepared datasets có sẵn:

```powershell
python report/source_latex/assets/render_report_figures.py
```

Các hình mới là tài nguyên báo cáo; notebook không phụ thuộc vào renderer này. Nguồn LaTeX và mọi assets đi cùng PDF. README/public sử dụng cùng thông tin nhóm và đường dẫn báo cáo tổng.

## Rà soát chuỗi mã và ký tự hiển thị

- Hai mã chia tập 64 ký tự ở Fashion-MNIST và GTSRB được đối chiếu đúng với `split_hash` trong metadata dataset, giữ nguyên tại `assets/split_identifiers.json`. PDF mô tả quy tắc chia tập và chỉ dẫn tới metadata ở A.10, không in các chuỗi mã dài.
- Tài liệu tham khảo GTSRB dùng tên liên kết “GTSRB Public Archive (ERDA)”; đích URL archive giữ nguyên.
- Rà toàn bộ 99 trang và nguồn LaTeX: không có ký tự thay thế U+FFFD, ký tự điều khiển bất thường, chuỗi hex dài hiển thị, tham chiếu `??`/`[?]` hoặc lỗi missing glyph. Ký hiệu toán co giãn được kiểm tra bằng bản render, không kết luận lỗi font chỉ từ các alias khi trích xuất text.
- Render trước/sau để đối chiếu bố cục; số liệu bảng và ảnh kết quả giữ nguyên byte. Notebook không sửa hoặc chạy lại.
