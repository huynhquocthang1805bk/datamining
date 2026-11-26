## 7. Minh chứng — giải thích kết quả từ cách mô hình hoạt động

### 7.1 Ảnh hưởng của thuộc tính lên mô hình dự báo

Chúng tôi đo **permutation importance trên tập test**: xáo trộn một nhóm đầu vào giữa các mẫu (giữ nguyên phân phối, cắt đứt liên hệ với mục tiêu) rồi xem RMSE tăng bao nhiêu. Với đầu vào không gian, phép đo làm riêng cho từng nút đích (40 nút ngẫu nhiên): xáo trộn chuỗi của chính nút / của các láng giềng 1-hop / của cùng số nút ở xa (≥ 4 hop, đối chứng).

![Hình 12. Ảnh hưởng của từng độ trễ (trái) và của từng nguồn thông tin (phải)](figures/fi_importance_losloop.png)

FI_TABLE

Đọc kết quả:

- **Độ trễ gần nhất áp đảo.** Với mọi mô hình, xáo trộn giá trị 5 phút trước làm RMSE tăng mạnh nhất; từ 15 phút trở về trước ảnh hưởng nhỏ dần. Điều này khớp với tự tương quan (0.82 ở 5 phút) và giải thích vì sao persistence mạnh. Riêng T-GCN gốc dựa nhiều hơn vào các độ trễ xa (30–55 phút) — dấu hiệu trạng thái ẩn bị "làm trơn" theo cả không gian lẫn thời gian.
- **Lịch sử của chính nút là nguồn thông tin chính (ΔRMSE ≈ 5.5–6.6).** T-GCN gốc là ngoại lệ: chỉ 3.6 cho chính nút (các mô hình khác 5.5–6.6) và tới 2.1 cho láng giềng — một phần lớn thông tin bị chuyển từ chính nút sang láng giềng, đúng như dự đoán từ trọng số của Â.
- **Láng giềng xa có ảnh hưởng đúng bằng 0** với mọi mô hình (kiểm tra hợp lệ của phương pháp); với GRU, láng giềng 1-hop cũng bằng 0 — đúng thiết kế.
- **Đặc trưng thời gian (giờ, thứ) có ích rõ** cho các mô hình RT (ΔRMSE ≈ 1.6–1.8), chủ yếu ở tầm xa (mục 5.4).

### 7.2 Trường tiếp nhận không gian–thời gian

Gradient saliency |∂ŷ_i/∂x_j(t−ℓ)| (dự báo 15 phút) gộp theo khoảng cách đồ thị giữa i và j và theo độ trễ ℓ cho thấy mô hình *thực sự* nhìn vào đâu:

![Hình 13. Độ nhạy của dự báo theo khoảng cách trên đồ thị và độ trễ](figures/fi_saliency_losloop.png)

- **GRU** chỉ nhạy với chính nút (hàng 0 hop), giảm dần theo độ trễ.
- **T-GCN gốc**: độ nhạy với chính nút ở độ trễ 5 phút giảm từ ≈ 0.41 (GRU) xuống ≈ 0.27, phần còn lại chuyển sang láng giềng 1–2 hop. Vì mỗi bước GRU áp Â lên cả đầu vào lẫn trạng thái ẩn, trường tiếp nhận về lý thuyết mở rộng thêm ít nhất 1 hop mỗi bước; thực tế mô hình chỉ dùng tới ~2 hop.
- **T-GCN tách self/láng giềng** khôi phục độ nhạy với chính nút (≈ 0.40) mà vẫn giữ một phần thông tin 1-hop — đúng như thiết kế.

### 7.3 Vì sao T-GCN không thắng GRU: tín hiệu của chính nút bị "loãng"

Với Â = D^(-1/2)(A + I)D^(-1/2), giá trị mà nút i "thấy" ở mỗi bước là Σ_j Â_ij x_j. Trên Los-loop (bậc trung bình 12.7, cạnh có trọng số), **phần trọng số dành cho chính nút chỉ là 32% (trung vị 26%)** với chuẩn hóa của tác giả, và 20.5% nếu self-loop = 1. Nghĩa là ~70% "đầu vào" của mỗi nút là tốc độ của láng giềng. Muốn dự báo y_{t+5'} ≈ x_t (điều persistence làm được), mạng phải tự "gỡ trộn" — điều một phép biến đổi tuyến tính dùng chung cho mọi nút không làm được chính xác. Bốn bằng chứng độc lập:

1. **Liều–đáp ứng** (cùng seed 42): self-loop 1 → 2 (tỉ trọng tự thân 20% → 32%) đưa RMSE 5 phút của T-GCN từ 5.96 về 4.84; tách hẳn self ([x, Âx]W, tỉ trọng 100%) đưa về 4.23 = GRU.
2. **Permutation importance** (7.1): T-GCN gốc dùng lịch sử chính nút ít hơn hẳn các mô hình khác.
3. **Bảng lỗi theo tầm:** T-GCN kém ở tầm ngắn (cần chính xác giá trị hiện tại), tốt hơn ở tầm xa (cần xu hướng khu vực, nơi trung bình láng giềng là thông tin tốt) — đúng chữ ký của một bộ lọc thông thấp không gian.
4. **Sửa đúng chỗ thì hết lỗi:** chỉ thay Âx bằng [x, Âx] (giữ nguyên mọi thứ khác), T-GCN vượt cả T-GCN gốc (p < 0.001) lẫn GRU (p = 0.01) qua 3 seed — thông tin láng giềng có giá trị thật, vấn đề nằm ở *cách trộn*. Trên SZ-taxi, hệ quả còn rõ hơn: T-GCN không "triệt tiêu" được phần residual x_t nên T-GCN-RT tệ hẳn trong khi GRU-RT không bị ảnh hưởng (mục 10).

Điều này cũng giải thích **P5** (MAE của GRU luôn thấp hơn T-GCN trong Bảng 1 bài báo): trung bình hóa theo láng giềng làm dự báo trơn hơn, giảm sai số lớn (RMSE) nhưng tăng sai số điển hình (MAE).

### 7.4 Vì sao residual và đặc trưng thời gian có ích

- **Skill so với persistence** (Hình 14): GRU-RT +9% ở 5 phút → +40% ở 60 phút; T-GCN gốc −19% ở 5 phút. Residual biến bài toán thành "dự báo phần thay đổi", nên mô hình mặc định đã ≥ persistence; thời gian cho biết xu hướng sắp tới (sắp vào/ra giờ cao điểm), thứ mà 60 phút lịch sử không chứa.
- **Mô hình dự báo được bao nhiêu phần thay đổi?** Hồi quy thay đổi dự báo theo thay đổi thực (Hình 15): hệ số góc chỉ 0.22 (15 phút) và 0.41 (60 phút) với T-GCN-RT; tương quan 0.38 và 0.59. Mô hình chỉ dám dự báo 20–40% biên độ thay đổi thật — đặc trưng của dự báo kỳ vọng có điều kiện dưới loss MSE khi tương lai bất định (sắp kẹt hay không).
- **Cổng cập nhật u của GRU bên trong T-GCN** (Hình 16) phản ứng đúng với trạng thái giao thông: u (giữ bộ nhớ) cao khi thông thoáng (tương quan Spearman +0.47 với tốc độ hiện tại) và giảm khi vừa có biến động lớn (−0.13 với |Δ| gần nhất) — mô hình "ghi đè" bộ nhớ bằng quan sát mới khi tình huống thay đổi. Ở T-GCN-RT, u thấp hơn hẳn (≈ 0.45 so với 0.78): vì phần "giữ giá trị cũ" đã có residual lo, trạng thái ẩn chỉ còn mã hóa thay đổi.

![Hình 14. Skill của từng mô hình so với persistence theo tầm dự báo](figures/exp_skill_losloop.png)

![Hình 15. Thay đổi dự báo so với thay đổi thực tế (T-GCN-RT)](figures/exp_delta_tgcn_res_time_losloop.png)

![Hình 16. Cổng cập nhật u theo biến động gần nhất và theo tốc độ](figures/exp_gates_losloop.png)

### 7.5 Vì sao dữ liệu này hạn chế lợi ích của đồ thị

Ngay ở bước EDA (mục 2.2), tương quan của *biến động* 5 phút giữa hai nút kề chỉ 0.066: láng giềng cho biết mức tốc độ chung của khu vực nhưng gần như không báo trước dao động kế tiếp. Cộng với chỉ 5 ngày train, mô hình có thêm thông tin không gian dễ học các mẫu trùng hợp hơn là quy luật (T-GCN-RT tách self dừng sớm ở epoch 18; T-GCN+ không học được gì từ đồ thị thích nghi). Đồ thị chỉ có lợi rõ ràng ở các sự kiện biến động mạnh (mục 5.3) và ở tầm xa.
