## 1. Ý tưởng

**Bài toán.** Cho tốc độ quan sát được tại N cảm biến (Los-loop: 207 vòng từ trên cao tốc Los Angeles, 5 phút/mốc; SZ-taxi: 156 tuyến đường quận La Hồ – Thâm Quyến, 15 phút/mốc) trong L bước gần nhất, dự báo tốc độ của **toàn bộ mạng** trong H bước tiếp theo (15, 30, 45, 60 phút).

**Ý tưởng của T-GCN (Zhao et al., 2019).** Tốc độ ở một đoạn đường phụ thuộc vào (i) chính quá khứ của nó (phụ thuộc thời gian) và (ii) trạng thái các đoạn đường lân cận (phụ thuộc không gian: ùn tắc lan truyền ngược dòng). GRU mô hình hóa (i); GCN mô hình hóa (ii). T-GCN đặt GCN **vào bên trong các cổng của GRU**: ở mỗi bước, trước khi quyết định "nhớ" hay "ghi đè", mỗi nút tổng hợp thông tin của các nút kề.

**Câu hỏi nghiên cứu của đồ án.** Chúng tôi không chỉ chạy lại mà kiểm chứng bốn giả thuyết:

- **H1 – Đồ thị có ích:** T-GCN tốt hơn GRU (cùng dữ liệu, cùng chia tập, cùng seed).
- **H2 – Học sâu có ích:** GRU/T-GCN tốt hơn các baseline đơn giản, kể cả baseline "giữ nguyên giá trị cuối" (persistence) mà bài báo không so sánh.
- **H3 – Con số của bài báo đáng tin:** giao thức đánh giá của tác giả không làm lạc quan kết quả.
- **H4 – Có thể cải tiến:** điều chỉnh cửa sổ đầu vào, số hidden units, thêm đặc trưng thời gian và học phần thay đổi (residual) giúp giảm sai số, nhất là giờ cao điểm.

## 2. Dẫn dắt — dữ liệu và kiểm tra dữ liệu (giai đoạn 1)

### 2.1 Tổng quan hai bộ dữ liệu

| Thuộc tính | Los-loop | SZ-taxi |
|---|---|---|
| Số mốc × số nút | 2016 × 207 | 2976 × 156 |
| Bước thời gian | 5 phút | 15 phút |
| Khoảng thời gian | 01/03/2012 (thứ Năm) → 07/03/2012 | 01/01/2015 → 31/01/2015 |
| Tốc độ TB ± độ lệch chuẩn | 58.9 ± 12.5 (mph) | 12.2 ± 12.7 |
| Giá trị 0 | 0% | **24.8%** (37 tuyến có > 50% là 0) |
| Đồ thị | 2626 cạnh, bậc TB 12.7, đối xứng, có trọng số theo khoảng cách | 532 cạnh, bậc TB 3.4, **có hướng**, nhị phân |
| Train / Val / Test (70/10/20 theo thời gian) | đến 05/03 21:35 / đến 06/03 14:20 / phần còn lại | đến 22/01 16:45 / đến 25/01 19:00 / phần còn lại |

Dữ liệu không có cột thời gian; chúng tôi dựng mốc thời gian từ ngày bắt đầu theo bài báo. **Kiểm chứng**: hồ sơ tốc độ trung bình theo giờ (Hình 1) cho thấy Los-loop có hai đáy rõ rệt lúc 7–9h và 16–19h vào ngày thường, còn cuối tuần gần như không có — đúng với giờ cao điểm địa phương, nên mốc thời gian dựng lại là hợp lệ và các đặc trưng giờ/thứ có nghĩa.

![Hình 1. Hồ sơ tốc độ theo giờ: ngày thường vs cuối tuần](figures/eda_daily_profile.png)

![Hình 2. Chuỗi tốc độ của 3 cảm biến Los-loop và cách chia train/val/test theo thời gian](figures/eda_timeline_losloop.png)

![Hình 3. Bản đồ nhiệt tốc độ (cảm biến × thời gian) Los-loop: các "sóng" kẹt xe xuất hiện đồng thời ở nhiều cảm biến vào giờ cao điểm](figures/eda_heatmap_losloop.png)

### 2.2 Bằng chứng cho phụ thuộc thời gian và không gian

- **Thời gian (Hình 4):** tự tương quan trung bình của Los-loop là 0.82 ở độ trễ 5 phút, 0.74 ở 15 phút và còn 0.50 ở 60 phút ⇒ cửa sổ 12 bước (60 phút) chứa phần lớn thông tin hữu ích; thông tin gần nhất quan trọng nhất. SZ-taxi chỉ 0.41 ở 15 phút và giảm rất chậm ⇒ chuỗi rất **nhiễu** (tốc độ ước lượng từ một số ít taxi), trung bình trượt sẽ là baseline mạnh.
- **Không gian (Hình 5):** hệ số tương quan giữa hai cảm biến giảm theo khoảng cách trên đồ thị: Los-loop 0.41 (kề trực tiếp) → 0.24 (2 bước) → 0.11 (≥ 10 bước). Đây là cơ sở cho GCN. Tuy nhiên, tương quan của **biến động ngắn hạn** (Δ tốc độ 5 phút) giữa hai nút kề chỉ 0.066 (và ≈ 0 với nút xa): láng giềng cho biết "mức" tốc độ chung của khu vực, nhưng rất ít thông tin về cú sốc kế tiếp. Ta kỳ vọng lợi ích của đồ thị là **có nhưng nhỏ** ở tầm ngắn.

![Hình 4. Tự tương quan theo độ trễ](figures/eda_autocorr.png)

![Hình 5. Tương quan giữa các cặp cảm biến theo khoảng cách trên đồ thị](figures/eda_corr_vs_hop.png)

### 2.3 Chất lượng dữ liệu — những điều bài báo không nói

1. **Los-loop chỉ có 7 ngày.** Tập train chứa 5 ngày (trong đó có 2 ngày cuối tuần), tức chỉ khoảng 3 lần quan sát mỗi khung giờ cao điểm ngày thường. Mọi mô hình học "theo giờ" đều có nguy cơ quá khớp.
2. **Dữ liệu nội suy.** Bài báo cho biết đã nội suy tuyến tính chỗ thiếu. Chúng tôi phát hiện 3.5% điểm nằm trên các đoạn thẳng tuyệt đối (≥ 6 điểm thẳng hàng), trong đó có một sự cố **mất dữ liệu toàn mạng** ngày 05/03 từ 13:30 đến 14:55 (100% cảm biến bị nội suy). Tập test có 4.6% mục tiêu như vậy; dự báo "trơn" sẽ được thưởng một cách giả tạo trên các điểm này (kiểm tra ở mục 8.5).
3. **SZ-taxi: số 0 nghĩa là "không có dữ liệu".** 24.8% giá trị bằng 0; tỉ lệ 0 cao nhất lúc 5–7h sáng (khi ít taxi), không phải giờ cao điểm. Đó là thiếu quan sát chứ không phải kẹt xe (Hình 6). Bài báo vẫn huấn luyện và chấm điểm trên các số 0 này. Chúng tôi báo cáo cả hai cách chấm (gồm 0 – để so với bài báo, và bỏ 0 – có ý nghĩa thực tế).

![Hình 6. SZ-taxi: tỉ lệ giá trị 0 trên mỗi tuyến và theo giờ](figures/eda_sz_zeros.png)
