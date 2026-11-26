## 9. So sánh với bài báo

### 9.1 Cùng cách tính của bài báo ("15 phút" = trung bình các bước 1…k)

Los-loop, giao thức nghiêm ngặt của chúng tôi (chọn epoch bằng validation), so với Bảng 1 của Zhao et al. (chọn epoch bằng test). Mỗi ô: RMSE / MAE / Accuracy / R².

PAPER_TABLE

- **HA, GCN, GRU tái lập sát bài báo**; HA trung bình cửa sổ của chúng tôi gần như trùng HA của tác giả ở "15 phút" (7.49 / 3.97 so với 7.44 / 4.01), xác nhận cách hiểu "HA = trung bình 12 bước đầu vào".
- **Hồi quy tuyến tính của chúng tôi tốt hơn SVR của bài báo** (5.49 so với 6.01 ở 15 phút) — baseline của bài báo yếu hơn mức có thể đạt.
- **T-GCN của chúng tôi kém con số bài báo** (5.55 so với 5.13 ở 15 phút; 7.52 so với 7.27 ở 60 phút). Mục 9.2 kiểm tra xem bao nhiêu phần đến từ giao thức.
- **Mô hình cải tiến (ensemble GRU-RT) vượt mọi con số của bài báo**: "15 phút" 5.09 / 3.01 (bài báo T-GCN 5.13 / 3.18), "60 phút" 6.91 / 4.01 (7.27 / 4.60) — dưới giao thức chặt hơn và ít dữ liệu train hơn.

### 9.2 Tái lập đúng giao thức của tác giả

Chúng tôi chạy lại GRU và T-GCN theo **đúng giao thức trong mã của tác giả** (`--protocol paper`): chia 80/20 không có validation, chuẩn hóa chia cho max của toàn chuỗi, cửa sổ test nằm trọn trong 20% cuối, lr 10⁻³ cố định, batch 32, hidden 64 (Los-loop) / 100 (SZ-taxi), dự báo pre_len = 3 bước (Los-loop, "15 phút") và 1 bước (SZ-taxi), **giữ epoch có RMSE test nhỏ nhất**. Do giới hạn CPU, chúng tôi train 300 epoch (Los-loop) và 150 epoch (SZ-taxi) — 5–10% so với 3000 epoch của tác giả.

PAPER_PROTOCOL_TABLE

![Hình 19. RMSE test theo epoch dưới giao thức của tác giả; đường chấm: con số trong bài báo; đường gạch: persistence trên đúng các cửa sổ test đó](figures/paper_protocol_curves.png)

Nhận xét:

- **GRU tái lập được bài báo ngay cả với ít epoch** (bão hòa sớm). **T-GCN hội tụ rất chậm** dưới chuẩn hóa max và lr cố định: Los-loop 6.01 (epoch 100) → 5.52 (200) → 5.36 (300), vẫn đang giảm; với 3000 epoch có thể tiến tới con số 5.13 của bài báo. Chúng tôi không ngoại suy; điều chắc chắn là ở cùng ngân sách huấn luyện, T-GCN chưa vượt GRU.
- **Ngay cả dưới giao thức của tác giả, MAE của T-GCN (3.25) vẫn kém persistence (3.16)** trên đúng các cửa sổ test đó — điểm P4 không phụ thuộc vào cách chúng tôi huấn luyện.
- Trên SZ-taxi, GRU tái lập sát bài báo (4.10 / 2.74 so với 4.05 / 2.68) ngay từ những epoch đầu, còn T-GCN sau 150 epoch (4.40, epoch tốt nhất là epoch cuối) vẫn kém trung bình trượt 3 giờ trên cùng cửa sổ (4.17 / 2.74). Với chuỗi nhiễu như SZ-taxi, gần như toàn bộ "kỹ năng" mà bài báo báo cáo đạt được bằng một trung bình trượt (P6b).
- Giao thức nghiêm ngặt của chúng tôi (z-score theo train, lr cosine, dừng sớm bằng validation) cho T-GCN hội tụ trong ~100 epoch; đây là lý do chúng tôi dùng nó cho mọi so sánh chính.

### 9.3 Tóm tắt các giả thuyết

| Giả thuyết | Kết luận | Bằng chứng chính |
|---|---|---|
| H1 – Đồ thị có ích (T-GCN > GRU) | **Bác bỏ ở dạng gốc; đúng sau khi sửa** | T-GCN gốc ≈ GRU (p = 0.76), MAE kém hơn; dạng tách self/láng giềng tốt hơn GRU có ý nghĩa (p = 0.01); đồ thị giúp ở tầm xa, biến động mạnh, dữ liệu nhiễu — nhưng không cộng thêm khi đã có residual + thời gian |
| H2 – Học sâu > baseline đơn giản | **Đúng tổng thể, sai ở giờ cao điểm** | Mọi mô hình vượt persistence (p < 0.005), nhưng trong giờ cao điểm persistence có MAE thấp nhất |
| H3 – Con số bài báo đáng tin | **Lạc quan** | Chọn epoch theo test, gộp tầm dự báo, không baseline persistence, không độ biến thiên |
| H4 – Có thể cải tiến | **Đúng** | Residual + thời gian (+ ensemble 3 seed): −7.8% RMSE so với GRU, −8.0% so với T-GCN, có ý nghĩa thống kê; tách self/láng giềng giúp T-GCN |

## 10. Kiểm chứng trên SZ-taxi

Chúng tôi lặp lại giai đoạn 1–2 (3 seed) và các cải tiến chính (1 seed) trên SZ-taxi — bộ dữ liệu khác hẳn Los-loop: đường đô thị, tốc độ ước lượng từ taxi (rất nhiễu, tự tương quan 15 phút chỉ 0.41), 24.8% giá trị bằng 0 (thiếu quan sát), đồ thị có hướng và thưa (bậc 3.4). Hidden = 64, L = 12 (3 giờ), H = 4 (15…60 phút).

**Tất cả mục tiêu (cách chấm của bài báo, gồm cả số 0):**

SZ_MAIN_TABLE

*RMSE validation của hai biến thể "loss bỏ qua 0" được tính trên mục tiêu ≠ 0 nên không so trực tiếp với các dòng khác.*

**Chỉ mục tiêu ≠ 0 (có ý nghĩa thực tế):**

SZ_MASKED_TABLE

Nhận xét:

- **Cùng kết luận với Los-loop: T-GCN không vượt GRU** — trên SZ-taxi còn kém hơn có ý nghĩa (mục tiêu ≠ 0: 4.69 so với 4.62; kiểm định cặp trên sai số trung bình 3 seed: ΔRMSE +0.07, KTC 95% [+0.06; +0.09], DM p < 10⁻⁴). Theo validation, mô hình cuối cho SZ-taxi là **ensemble 3 seed GRU** (RMSE val 3.970, GRU-RT 3.977).
- **Học sâu chỉ hơn trung bình trượt rất ít:** GRU 4.18 / 2.78 so với trung bình 12 bước đầu vào 4.24 / 2.79 (gồm số 0); 4.62 / 3.27 so với 4.74 / 3.37 (bỏ số 0). Persistence rất tệ (5.39) vì giá trị cuối nhiễu và hay bằng 0. Chuỗi SZ-taxi gần như "mức trung bình + nhiễu" ở độ phân giải 15 phút, nên trung bình hóa (theo thời gian hoặc theo láng giềng) là gần tối ưu.
- **Residual không giúp, thậm chí phá T-GCN:** GRU-RT = GRU (p = 0.57), còn T-GCN-RT tệ hẳn (4.79). Với chuỗi nhiễu, mô hình phải học *triệt tiêu* phần x_t được cộng vào (Δ̂ ≈ mức trung bình − x_t), điều cần biết chính xác x_t — GRU làm được, T-GCN chỉ thấy ~26% giá trị của chính nút sau Â nên không làm được. Đây là thêm một hệ quả của hiện tượng "loãng tín hiệu" (mục 7.3), và cho thấy cải tiến tốt trên Los-loop (chuỗi mượt, tự tương quan 0.82) **không tự động chuyển sang** dữ liệu nhiễu.
- **Loss bỏ qua số 0 không cải thiện** (T-GCN: 4.72 so với 4.69 trên mục tiêu ≠ 0), có lẽ vì đầu vào vẫn chứa số 0 và mô hình mất đi một phần tư tín hiệu huấn luyện. Cách xử lý đúng hơn là điền giá trị thiếu ở *đầu vào* (ví dụ bằng trung bình láng giềng hoặc theo giờ) — hướng phát triển.
- **Đồ thị lại có ích khi tốc độ biến động mạnh** (MAE 15 phút với |Δ| > 10: T-GCN 8.50, GCN 8.62 so với GRU 9.05, persistence 14.3) — đúng như Los-loop. Trên SZ-taxi không có giờ cao điểm rõ rệt (MAE cao điểm ≈ ngoài cao điểm).
- Dưới giao thức của tác giả (mục 9.2), GRU tái lập sát bài báo (4.10 so với 4.05), T-GCN sau 150 epoch vẫn chưa hội tụ (4.40 so với 3.92).

