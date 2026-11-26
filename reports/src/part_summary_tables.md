## Bảng tổng hợp: mọi mô hình so với baseline gốc (T-GCN)

**Baseline gốc** là mô hình T-GCN của Zhao et al. (2019) — mô hình mà đề tài tái lập. Vì có hai cách chấm điểm nên có hai bảng cho mỗi bộ dữ liệu, mỗi bảng một baseline:

1. **Theo cách tính của bài báo** ("15/30/45/60 phút" = sai số gộp các bước 1…k; RMSE, MAE, Accuracy, R²). Baseline là **con số T-GCN in trong Bảng 1 của bài báo**. Bảng này trả lời câu hỏi "có vượt con số đã công bố không?" — lưu ý tác giả chọn epoch theo tập test và chia 80/20 (mục 4), nên đây là so với một mốc *lạc quan*.
2. **Theo giao thức nghiêm ngặt** (70/10/20, chọn epoch theo validation, 3 seed, sai số tại đúng từng bước). Baseline là **T-GCN gốc do chúng tôi tái lập** — cùng dữ liệu, cùng cách chia, cùng các cửa sổ test với mọi mô hình khác. Đây là so sánh công bằng.

**Quy ước cho mọi bảng so sánh trong báo cáo:** dòng baseline được đánh dấu *baseline*; **số in đậm = tốt hơn baseline ở cùng cột** (RMSE, MAE thấp hơn; Accuracy, R² cao hơn), so ở đúng số chữ số hiển thị nên hai số bằng nhau trên bảng không được in đậm; Δ = thay đổi tương đối so với baseline (âm = sai số giảm). Các dòng "(GPU)" là các lần chạy trên GPU máy của bạn (7 cấu hình × 3 seed × 2 bộ dữ liệu, mục 11.8); chỉ số "kiểu bài báo" của chúng được tính chính xác từ sai số từng bước (mục 11.8).

### Los-loop — cách tính của bài báo (baseline: T-GCN trong Bảng 1)

VS_PAPER_LL

### Los-loop — giao thức nghiêm ngặt (baseline: T-GCN gốc tái lập)

VS_STRICT_LL

### SZ-taxi — cách tính của bài báo (baseline: T-GCN trong Bảng 1)

VS_PAPER_SZ

### SZ-taxi — giao thức nghiêm ngặt (baseline: T-GCN gốc tái lập)

VS_STRICT_SZ

**Đọc nhanh bốn bảng:**

- **Los-loop, so với con số của bài báo:** năm mô hình tốt hơn T-GCN của bài báo ở **cả 16/16 ô** (4 tầm × 4 chỉ số): mô hình cuối (trộn theo tầm), LightGBM, và cả ba bản Graph WaveNet + residual (CPU, GPU, GPU cỡ đầy đủ); ensemble GRU-RT đạt 15/16 (Accuracy "15 phút" bằng nhau, 0.913). Mô hình cuối giảm **14.8% RMSE và 24.1% MAE** ở "60 phút" (6.190 / 3.494 so với 7.268 / 4.602). Ngược lại, T-GCN gốc tái lập dưới giao thức nghiêm ngặt **không đạt** con số RMSE của bài báo (5.552 so với 5.126 ở "15 phút"; mục 9.2 cho thấy phần lớn chênh lệch đến từ giao thức), chỉ tốt hơn về MAE ở 45–60 phút.
- **Hai dòng tự nó là phản biện:** chính GRU *trong bài báo* và persistence (giữ giá trị cuối, không học gì) đều có **MAE tốt hơn T-GCN của bài báo ở cả 4 tầm** — kết luận "T-GCN tốt nhất trên mọi chỉ số" không đứng vững ngay trên Bảng 1 (mục 4, P4–P5).
- **Los-loop, giao thức nghiêm ngặt:** mọi cải tiến đều tốt hơn T-GCN gốc ở cả 10/10 ô, trừ STID (9/10), STAEformer cỡ đầy đủ chạy GPU (7/10 — thua ở 5 phút và RMSE 15 phút) và DCRNN-lite (5/10, RMSE TB kém hơn 0.6%). Mô hình cuối: **−17.6% RMSE, −20.1% MAE**; mạng nơ-ron tốt nhất là Graph WaveNet + residual (−12.3% / −15.9% trên CPU; −12.7% / −14.4% trên GPU).
- **SZ-taxi, so với con số của bài báo:** khó vượt hơn nhiều vì chuỗi gần như "mức trung bình + nhiễu". Không mô hình nào có RMSE "15 phút" thấp hơn 3.916 của bài báo; chỉ mô hình cuối tốt hơn ở RMSE "60 phút" (4.008 so với 4.014, −0.2%). Nhưng Graph WaveNet / STAEformer chạy GPU và mô hình cuối tốt hơn ở **MAE và R² của cả 4 tầm** (8–9/16 ô; Graph WaveNet: −3.9% MAE). Accuracy (= 1 − ‖Y − Ŷ‖/‖Y‖) và R² (= 1 − SSE/SST) đều phụ thuộc *tập test* qua ‖Y‖ và phương sai của Y, mà tập test của bài báo (80/20) khác của chúng tôi; vì vậy trên SZ-taxi hai chỉ số này còn cho kết luận ngược nhau (R² tốt hơn, Accuracy kém hơn) — chúng chỉ so chặt chẽ được trên cùng tập test như ở bảng dưới.
- **SZ-taxi, giao thức nghiêm ngặt:** mọi mô hình học sâu và LightGBM tốt hơn T-GCN gốc ở 8/8 ô; **Graph WaveNet (GPU) là mô hình đơn tốt nhất (RMSE ≠0 4.428, −5.7%)**, nhỉnh hơn STID (4.445) nhưng chênh lệch (0.017) chưa tới 2 lần độ lệch chuẩn giữa các seed nên chưa đủ để kết luận (mục 11.8); mô hình cuối −6.2% RMSE, −7.2% MAE.
