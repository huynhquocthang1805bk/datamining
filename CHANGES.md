# Thay đổi so với T-GCN gốc và so với phiên bản đầu của dự án

## Trung thành với bài báo / mã tác giả
- Ô T-GCN: một lớp GCN trong mỗi cổng GRU, bias cổng khởi tạo 1.0, đầu ra tuyến tính cho H bước (như `tgcn.py` của tác giả).
- **Ma trận kề chuẩn hóa đúng như `calculate_laplacian` của tác giả**: Â = D^-1/2 (A+I)ᵀ D^-1/2, *không* đối xứng hóa, *cộng thêm* I dù `los_adj.csv` đã có đường chéo 1 (self-loop thực tế = 2). Phiên bản đầu của dự án đặt đường chéo = 1 và đối xứng hóa — giữ lại làm ablation (`variants/tgcn_selfloop1`), cho kết quả kém hơn rõ rệt.
- GRU baseline = cùng ô với Â = I (một bộ trọng số chung cho mọi nút), đúng như baseline của tác giả.
- `--protocol paper` tái lập nguyên vẹn giao thức của tác giả (80/20, chia max toàn chuỗi, cửa sổ test nằm trong 20% cuối, giữ epoch có RMSE test nhỏ nhất) — chỉ dùng để so sánh với Bảng 1.

## Giao thức đánh giá chính (nghiêm ngặt hơn bài báo)
- Chia 70/10/20 theo thời gian; z-score theo từng cảm biến bằng thống kê của train; dừng sớm theo validation; 3 seed.
- Sai số tại đúng bước k (per-step) **và** kiểu gộp 1..k của bài báo (cumulative); thêm Accuracy, R², Var như bài báo; bản bỏ số 0 cho SZ-taxi.
- RMSE test theo epoch được *ghi* (không dùng để chọn) để đo độ lạc quan của việc chọn theo test.

## Mô hình / tùy chọn mới
- `gcn`: baseline GCN 2 lớp (chỉ không gian) như bài báo.
- `--residual`, `--time-features` cho cả GRU và T-GCN; `--self-sep` ([x, Âx]W); `--no-graph` (Â = I); `--raw-loss`; `--loss mae`; `--mask-zero`.
- Baseline cổ điển (`baselines.py`): persistence, HA cửa sổ (đúng định nghĩa của tác giả), HA theo giờ, hồi quy tuyến tính.
- `predict.py`: nạp checkpoint tự chứa (trọng số + cấu hình + thống kê chuẩn hóa), ensemble nhiều checkpoint, cắt dự báo âm.

## Kết luận đã được thực nghiệm kiểm chứng (Los-loop, 3 seed)
- Residual + đặc trưng giờ/thứ: cải thiện có ý nghĩa (GRU-RT 6.98 so với GRU 7.50, p < 0.001).
- Tách self/láng giềng: T-GCN 7.30 so với GRU 7.50 (p = 0.01).
- Không có lợi / không ổn định: đồ thị thích nghi (T-GCN+), loss đơn vị gốc (7.14 ± 0.24), cửa sổ 24 bước, hidden 100. Huber đổi RMSE lấy MAE.

## Cải tiến nâng cao (đợt 2)
- `tgcn/models_modern.py`: STID, Graph WaveNet (cài đặt kênh-cuối, nhanh trên CPU), DCRNN-lite (diffusion hai chiều K = 2, đầu ra trực tiếp), STAEformer; cấu hình `compact` (CPU) và `full` (kích thước bài báo, GPU); tùy chọn residual.
- `gbdt.py`: LightGBM với ~50 đặc trưng không gian–thời gian (độ trễ, thống kê trượt, láng giềng qua Â, lịch, hồ sơ theo giờ của từng cảm biến; v2: bất thường láng giềng/toàn mạng), một mô hình cho mỗi tầm, dừng sớm trên validation; lưu/nạp mô hình (`predict.py` nhận thư mục).
- `--profile` cho GRU/T-GCN: hồ sơ theo giờ (chỉ từ train) làm đầu vào và thay đổi hồ sơ dự kiến tới t+k ở đầu ra.
- `analysis/blend.py`: trộn theo từng tầm, trọng số không âm tối ưu trên validation.
- `tools/prepare_large_dataset.py`, `LOCAL_RUN.md`: chạy trên GPU và METR-LA / PEMS-BAY.
- Kết quả (3 seed): trộn LightGBM + Graph WaveNet-res 6.19 / 3.49 trên Los-loop (−18% / −20% so với T-GCN); STID tốt nhất trên SZ-taxi.

## Kết quả chạy GPU và bảng so với baseline gốc
- Kết quả 42 run GPU của người dùng (Graph WaveNet, STAEformer; cỡ gọn/đầy đủ; có/không residual; 3 seed; Los-loop và SZ-taxi) được giữ nguyên trong `reports/external/` và gộp vào `all_runs_*.csv`, bảng mô hình hiện đại và bảng tổng hợp.
- `analysis/gpu_results.py`: kiểm tra số liệu GPU (tóm tắt của người dùng khớp tuyệt đối với 21 run; sai số tăng theo tầm; CPU ↔ GPU), đếm tham số, suy ra chính xác chỉ số kiểu bài báo (RMSE/MAE/Accuracy/R² gộp 1..k) từ sai số từng bước.
- `analysis/baseline_compare.py`: mọi mô hình so với baseline gốc — T-GCN trong Bảng 1 của bài báo (cách tính của bài báo) và T-GCN gốc tái lập (giao thức nghiêm ngặt).
- `tools/build_report.py`: mọi bảng so sánh có dòng *baseline* (tô nền trong bản Word) và **in đậm mọi số tốt hơn baseline** ở cùng cột, so ở đúng số chữ số hiển thị (`tests/test_report_bold.py`).
- Sửa: bảng 5.2 trước đây lấy cột "5 phút" từ seed 42 trong khi các cột khác là trung bình 3 seed — nay mọi cột là trung bình 3 seed (T-GCN 5 phút: 4.86, không phải 4.84); SZ-taxi chỉ có 1 seed GRU-RT nên "ensemble GRU-RT" không còn được liệt kê như một mô hình riêng và kiểm định ghi đúng "GRU-RT (1 seed)"; STID trên SZ-taxi làm tròn 2 chữ số là 4.44 (không phải 4.45).
