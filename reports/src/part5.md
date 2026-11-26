## 8. Đánh giá độ tin cậy của mô hình

### 8.1 Độ ổn định theo seed

SEED_TABLE

Độ lệch chuẩn giữa các seed nhỏ (CV ≤ 1%) với mọi cấu hình chính — ngoại lệ duy nhất là GRU-RT với loss đơn vị gốc (CV 3.4%, một seed kém hẳn), đó là lý do biến thể này bị loại. Các độ lệch này nhỏ, nhưng **cùng cỡ với nhiều khác biệt mà bài báo dùng để kết luận** (ví dụ T-GCN hơn GRU 1.8% RMSE ở "15 phút"). Ensemble 3 seed luôn cải thiện nhẹ và có ý nghĩa (GRU-RT: 6.98 → 6.91, DM p < 10⁻⁴).

### 8.2 Khác biệt nào có ý nghĩa thống kê?

Kiểm định cặp trên cùng các mẫu test: bootstrap khối dịch chuyển (khối 12 mẫu = 1 giờ, giữ tự tương quan) cho KTC 95% của ΔRMSE, và kiểm định Diebold–Mariano (phương sai Newey–West). Sai số mỗi mẫu được lấy trung bình qua 3 seed.

PAIRWISE_TABLE

Kết luận: (i) mọi mô hình học sâu vượt persistence có ý nghĩa; (ii) **T-GCN gốc và GRU không khác nhau có ý nghĩa** (p = 0.76), nhưng **T-GCN tách self/láng giềng tốt hơn cả hai** (p = 0.010 và p < 0.001); (iii) residual + thời gian cải thiện GRU có ý nghĩa (p < 0.001); (iv) trong cấu hình RT, GRU-RT tốt hơn cả T-GCN-RT (p = 0.003) lẫn T-GCN-RTS (p = 0.002); (v) ensemble hỗn hợp GRU-RT + T-GCN-RTS không hơn ensemble GRU-RT (p = 0.50).

Lưu ý: tập test của Los-loop chỉ là ~1.4 ngày (2 đợt cao điểm chiều, 1 đợt sáng). Các KTC trên phản ánh bất định *trong* giai đoạn đó, không bảo đảm cho tuần khác, mùa khác hay ngày lễ.

### 8.3 Độ bền khi dữ liệu đầu vào xấu

![Hình 17. RMSE khi thêm nhiễu Gauss (trái) và khi một tỉ lệ đầu vào bị mất, đặt = 0 (phải)](figures/rel_robustness_losloop.png)

- **Nhiễu đo đạc:** mọi mô hình suy giảm chậm hơn persistence. Trong nhóm không residual, **các mô hình có đồ thị bền hơn GRU** (σ = 8 mph: T-GCN 8.20, T-GCN tách self 8.32, GRU 8.58): tích chập đồ thị trung bình hóa nhiễu độc lập giữa các cảm biến — một lợi ích thật của đồ thị mà sai số trên dữ liệu sạch không cho thấy.
- **Cảm biến hỏng (đầu vào = 0):** các mô hình residual **rất mong manh** (5% đầu vào hỏng: T-GCN-RT 7.35 → 12.0; GRU-RT 7.05 → 9.4) vì chúng cộng thẳng giá trị cuối (= 0). GRU thường và T-GCN tách self bền nhất (5% hỏng: 8.39 và 8.37). ⇒ Khi triển khai mô hình residual **bắt buộc** phải kiểm tra/điền giá trị thiếu trước khi dự báo.

### 8.4 Khoảng dự báo có đáng tin không? (conformal)

Dùng phần dư trên validation để dựng khoảng dự báo 90% (split conformal, theo từng tầm; biến thể theo từng tầm × từng nút):

CONFORMAL_TABLE

Độ phủ thực tế 85–87% (thấp hơn mục tiêu 90%) và chỉ **69–79% trong giờ cao điểm** — vì phân phối sai số thay đổi giữa validation (sáng thứ Ba) và test (chiều thứ Ba–thứ Tư). Khoảng dự báo dùng được ngoài cao điểm (≈ 89–90%), nhưng cần hiệu chỉnh theo giờ (conformal có điều kiện) nếu dùng cho cảnh báo kẹt xe.

### 8.5 Các kiểm tra khác

- **Điểm nội suy:** loại 4.6% mục tiêu test nằm trên đoạn nội suy thẳng làm RMSE của mọi mô hình tăng ~0.06–0.10 (các điểm nội suy "dễ" hơn), thứ hạng không đổi.
- **Độ lệch theo mức tốc độ** (Hình 18): khi thực tế ≤ 20 mph, GRU/T-GCN dự báo cao hơn trung bình ~9 mph; khi thông thoáng (> 60 mph) thấp hơn ~1.4 mph. Mô hình "co về trung bình"; độ tin cậy thấp nhất đúng lúc cần nhất (kẹt xe).
- **Độ lạc quan của việc chọn epoch theo test:** trung bình 1.1% RMSE trên 37 run; 0.3–0.5% với GCN/GRU/T-GCN nhưng 2–5.6% với các biến thể dễ quá khớp (mục 4.1).

![Hình 18. Sai số trung bình theo khoảng tốc độ thực tế (dự báo 15 phút)](figures/rel_bias_by_level_losloop.png)
