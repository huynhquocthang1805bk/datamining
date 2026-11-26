## 5. Kết quả thực nghiệm trên Los-loop

Mọi con số dưới đây là trên **tập test** (20% cuối, 06/03 14:20 → 07/03 23:55), sai số tại đúng bước k (per-step) trừ khi ghi "kiểu bài báo"; "TB 12 tầm" là sai số gộp mọi bước 5…60 phút. Mô hình có 3 seed được ghi trung bình ± độ lệch chuẩn.

### 5.1 Giai đoạn 1 — GRU: kết quả ban đầu

GRU (hidden 64, L = 12) hội tụ ổn định (dừng sớm ở epoch 87–101) và **rất ít phụ thuộc seed**: RMSE TB 12 tầm = 7.495 ± 0.006, MAE = 4.175 ± 0.023. Tính theo đúng cách của bài báo, GRU của chúng tôi đạt RMSE 5.217 / MAE 2.965 ở "15 phút" và 7.495 / 4.175 ở "60 phút" — **khớp sát** GRU trong Bảng 1 của bài báo (5.218 / 3.060 và 7.662 / 4.519; RMSE "15 phút" gần như trùng), dù chúng tôi train trên ít dữ liệu hơn (70% so với 80%) và chọn epoch bằng validation. Kết luận giai đoạn 1: pipeline dữ liệu, chuẩn hóa và đánh giá là đúng; GRU là baseline đáng tin cậy và **vượt persistence ở mọi tầm** (skill +9% ở 5 phút, +26% ở 60 phút).

### 5.2 Giai đoạn 2 — T-GCN trên cùng dữ liệu, cùng cách chia

PHASE2_TABLE

Ba quan sát:

1. **T-GCN không tốt hơn GRU tổng thể.** Chênh lệch RMSE +0.02 (KTC 95% bootstrap khối [−0.12; +0.15], kiểm định Diebold–Mariano p = 0.76) — không có ý nghĩa thống kê; MAE thì GRU tốt hơn rõ (4.18 so với 4.37). Điều này **trùng khớp với Bảng 1 của chính bài báo**, nơi MAE của GRU thấp hơn T-GCN ở cả 8 ô (mục 4, P5).
2. **Có điểm giao nhau theo tầm dự báo:** ở tầm ngắn T-GCN kém GRU rõ (5 phút: 4.86 so với 4.24 — kém cả persistence 4.45; trong bảng, persistence được in đậm ở 5 phút vì tốt hơn T-GCN), nhưng ở 45–60 phút T-GCN tốt hơn (60 phút: 9.13 so với 9.32). Thông tin láng giềng có ích để dự báo xa (ùn tắc lan truyền) nhưng làm hại dự báo gần — mục 7.3 giải thích cơ chế.
3. **Cách chuẩn hóa đồ thị quyết định T-GCN tốt hay tệ.** Phiên bản đầu của dự án đặt đường chéo Â = 1 (không cộng thêm I như mã tác giả) cho T-GCN RMSE 7.93 / MAE 4.74 (5 phút: 5.96!). Chỉ riêng việc tăng trọng số tự thân (self-loop = 2, đúng như mã gốc) đã đưa T-GCN về 7.52 / 4.37. Đây là bằng chứng trực tiếp đầu tiên cho giả thuyết "loãng tín hiệu của chính nút".

![Hình 7. Sai số theo từng tầm dự báo (dải mờ: min–max qua 3 seed)](figures/cmp_horizon_losloop.png)

### 5.3 Giai đoạn 3 — So sánh sai số và phân tích giờ cao điểm

**Sai số tập trung vào giờ cao điểm.** Giờ cao điểm ngày thường (7–9h, 16–19h) chiếm 24% mục tiêu của tập test nhưng MAE ở đó cao gấp 1.4–2 lần ngoài cao điểm (Hình 8). Bảng dưới tách tập test theo trạng thái giao thông (MAE, trung bình 3 seed):

PEAK_TABLE

Những điều bảng này cho thấy — và bài báo không phân tích:

- **Trong giờ cao điểm, persistence có MAE thấp nhất ở cả 15 lẫn 60 phút; khi kẹt xe (v < 40 mph) nó cũng tốt nhất ở 15 phút**, và ở 60 phút chỉ GRU-RT nhỉnh hơn nó (13.4 so với 13.6). Mô hình học sâu thắng tổng thể chủ yếu nhờ 76% thời gian ngoài cao điểm, nơi chúng làm trơn tốt dao động nhỏ.
- **Khi tốc độ biến động mạnh (|Δ| > 10 mph so với giá trị cuối, 7% mẫu), các mô hình có đồ thị tốt nhất ở 15 phút** (GCN 13.3, T-GCN 14.1 so với GRU 15.6 và persistence 18.6): láng giềng phía trước "báo trước" cú sốc. Đây là nơi duy nhất đồ thị tạo khác biệt rõ ở tầm ngắn; ở 60 phút, GRU-RT lại tốt nhất (15.0).
- Mọi mô hình học sâu **dự báo lạc quan khi kẹt xe**: với tốc độ thực ≤ 20 mph, sai số trung bình (bias) là +9 mph (Hình 18) — hệ quả của hàm mất mát MSE (dự báo kỳ vọng có điều kiện, kéo về mức thông thoáng chiếm đa số).

![Hình 8. MAE theo giờ trong ngày; vùng xám là giờ cao điểm](figures/peak_mae_by_hour_losloop.png)

![Hình 9. Dự báo trước 15 phút ở cảm biến 763995 qua hai đợt kẹt xe của tập test](figures/peak_example_losloop.png)

Hình 9 cho thấy thêm: mọi mô hình đều "trễ" khoảng một tầm dự báo khi kẹt xe bắt đầu (không thể biết trước một sự kiện chưa có dấu hiệu trong 60 phút đầu vào), và T-GCN gốc cho cả dự báo **phi vật lý** (tốc độ âm tới −27 mph) vì đầu ra tuyến tính không bị chặn.

### 5.4 Giai đoạn 3 — Thử cải thiện: cửa sổ đầu vào, hidden units, đặc trưng thời gian, residual

Theo yêu cầu đề tài, chúng tôi thử ba "núm vặn" trên T-GCN: độ dài cửa sổ đầu vào, số hidden units và đặc trưng thời gian; sau đó thử thêm các thay đổi về *cách đặt bài toán* (residual, hàm mất mát) và về *cách dùng đồ thị*. Tinh chỉnh dùng seed 42 (so với cấu hình mặc định cùng seed).

**Cửa sổ đầu vào và số hidden units:**

TUNING_TABLE

![Hình 10. Tinh chỉnh cửa sổ đầu vào và số hidden units (hàng trên RMSE, hàng dưới MAE)](figures/tuning_losloop.png)

- **Cửa sổ L:** với T-GCN gốc, L = 6 (30 phút) kém hơn L = 12 (RMSE 7.58 so với 7.48, chủ yếu ở 60 phút: 9.29 so với 9.07) — nửa giờ lịch sử không đủ thấy xu hướng. L = 24 (2 giờ) không cải thiện mà tăng gấp đôi thời gian huấn luyện; với T-GCN-RT còn kém hơn (7.53 so với 7.35) vì thông tin > 1 giờ trước gần như không liên quan (tự tương quan ở 120 phút chỉ 0.29) mà làm mô hình dễ quá khớp. MAE nhỏ nhất ở L = 12 cho cả hai mô hình.
- **Hidden units:** 32 ≈ 64 (7.47 so với 7.48) với ¼ số tham số (3 660 so với 13 452); 100 kém hơn một chút (7.53 / 4.43) và quá khớp sớm hơn. Khớp nhận xét của bài báo (sai số giảm rồi tăng theo số hidden), nhưng biên độ chỉ ~1%.
- **Đặc trưng thời gian:** xem bảng ablation dưới (RMSE −3.4% nhưng MAE +3.1% nếu thêm riêng lẻ).

⇒ **Siêu tham số không phải đòn bẩy chính**: trong khoảng thử, mọi thay đổi ≤ 1.3% — nhỏ hơn nhiều tác động của residual + thời gian (−6% với GRU). Lựa chọn L = 12, hidden = 64 của bài báo là hợp lý.

**Từng thành phần một (cùng seed 42, so với T-GCN gốc):**

ABLATION_TABLE

![Hình 11. Đóng góp của từng thành phần (thay đổi % RMSE/MAE so với T-GCN gốc, TB 12 tầm, seed 42)](figures/ablation_losloop.png)

Đọc bảng ablation theo cặp có kiểm soát:

- **Residual sửa tầm ngắn, đặc trưng thời gian sửa tầm dài.** T-GCN + residual (seed 42): 5 phút 4.84 → 4.33 nhưng 60 phút gần như không đổi; T-GCN + thời gian: 60 phút 9.07 → 8.53 nhưng 5 phút xấu đi. Kết hợp (T-GCN-RT) lấy được cả hai.
- **Cùng ô T-GCN-RT nhưng bỏ đồ thị (Â = I) lại tốt hơn có đồ thị** (7.09 so với 7.35). Với cách chuẩn hóa của tác giả, đồ thị vật lý gây hại khi mô hình đã có residual + thời gian.
- **Tách self/láng giềng ([x, Âx]W) là cách duy nhất làm đồ thị có lợi:** T-GCN tách self (không RT) đạt **7.30 ± 0.06 / 4.12 ± 0.02 qua 3 seed — tốt hơn GRU có ý nghĩa thống kê** (ΔRMSE −0.20, KTC 95% [−0.34; −0.07], DM p = 0.010) và tốt hơn T-GCN gốc (p < 0.001); T-GCN-RT tách self đạt 7.17 / 4.13 so với 7.35 / 4.32 của T-GCN-RT.
- **Loss theo đơn vị gốc** (nhân trọng số std² cho mỗi cảm biến, để cảm biến hay kẹt được coi trọng như khi chấm điểm) cải thiện GRU-RT ở seed 42 (6.94 / 4.03 so với 7.05 / 4.08) — nhưng **không lặp lại qua 3 seed** (7.14 ± 0.24 so với 6.98 ± 0.06) và không giúp T-GCN-RT. Đây là ví dụ cụ thể vì sao không được kết luận từ một lần chạy (đúng lỗi P7 của bài báo).
- **Huber** đổi RMSE lấy MAE (MAE 4.32 → 4.14, RMSE 7.35 → 7.39) — đúng như lý thuyết (Huber nhắm trung vị nhiều hơn kỳ vọng).
- **Đồ thị thích nghi (T-GCN+)** không mang lại gì (7.37 so với 7.35): 207 × 8 tham số embedding không học được quan hệ mới từ 5 ngày dữ liệu.

### 5.5 Mô hình cuối cùng — chọn bằng validation, không bằng test

Để không lặp lại lỗi chọn-theo-test của bài báo, cấu hình cuối được chọn theo **RMSE validation** (đơn vị gốc, so sánh được giữa các hàm mất mát), sau đó mới xem test:

FINAL_TABLE

- **Theo validation, GRU-RT là cấu hình tốt nhất** (RMSE val 5.906 ± 0.056); thứ hạng trên test giống hệt thứ hạng trên validation, nên lựa chọn này không "nhìn trộm" test. **Mô hình cuối cùng = ensemble 3 seed của GRU-RT**: test RMSE 6.91 / MAE 4.01 TB 12 tầm — giảm 7.8% RMSE và 3.9% MAE so với GRU, 8.0% / 8.2% so với T-GCN gốc (Bảng 9.1: tốt hơn mọi con số của bài báo).
- **Mô hình đồ thị tốt nhất là T-GCN tách self/láng giềng**: không residual, không đặc trưng giờ mà vẫn vượt GRU có ý nghĩa (7.30 so với 7.50, p = 0.01). Đây là câu trả lời cho câu hỏi "đồ thị có giúp không?": *có*, nếu phép tích chập không xóa mờ thông tin riêng của nút.
- Ensemble hỗn hợp GRU-RT + T-GCN-RTS có MAE thấp nhất (4.00) nhưng không khác ensemble GRU-RT có ý nghĩa (p = 0.50) mà tốn gấp đôi chi phí ⇒ không chọn.
- Hai khuyến nghị khi triển khai đi kèm mô hình cuối (mục 8): kiểm tra/điền giá trị thiếu trước khi dự báo (mô hình residual nhạy với đầu vào = 0) và cắt dự báo về miền vật lý [0, v_max] (đã cài trong `predict.py`).
