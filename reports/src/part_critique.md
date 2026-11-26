## 4. Phản biện bài báo

Các điểm dưới đây được kiểm chứng trực tiếp từ Bảng 1 của bài báo và mã nguồn chính thức (`lehaifeng/T-GCN/T-GCN/T-GCN-TensorFlow/main.py`, `input_data.py`), không phải suy đoán.

**P1. Chọn epoch theo tập test.** Mã của tác giả đánh giá tập test sau *mỗi* epoch và báo cáo kết quả ở epoch có RMSE test nhỏ nhất:
```python
index = test_rmse.index(np.min(test_rmse))      # main.py, dòng cuối
print('min_rmse:%r' % (np.min(test_rmse)), 'min_mae:%r' % (test_mae[index]), ...)
```
Không có tập validation. Đây là rò rỉ thông tin từ test vào lựa chọn mô hình: con số báo cáo là *cận dưới lạc quan*, không phải sai số kỳ vọng trên dữ liệu mới. Chúng tôi đo trực tiếp độ lạc quan này ở mục 4.1.

**P2. Chuẩn hóa bằng giá trị lớn nhất của toàn bộ chuỗi** (`max_value = np.max(data1)` trước khi chia train/test) — một thống kê của tập test được dùng khi huấn luyện. Ảnh hưởng nhỏ (Los-loop max = 70 ở cả train lẫn test) nhưng sai về nguyên tắc.

**P3. Cách gộp sai số theo tầm dự báo.** Mã tác giả làm phẳng mọi bước đầu ra (`reshape(test_output, [-1, num_nodes])`) trước khi tính chỉ số, nên "15 phút" trên Los-loop thực chất là **trung bình của 5, 10 và 15 phút**, "60 phút" là trung bình của 12 bước. Con số vì thế dễ hơn sai số thật tại đúng thời điểm 60 phút: với GRU của chúng tôi, RMSE tại đúng bước 60 phút là 9.32 nhưng kiểu gộp của bài báo chỉ 7.50 (thấp hơn 20%).

**P4. Thiếu baseline persistence.** Baseline "giữ nguyên giá trị cuối" (ŷ_{t+k} = x_t) không cần học gì, là thước đo tối thiểu trong dự báo giao thông. Trên Los-loop, dùng **đúng cửa sổ test và cách gộp của tác giả** (80/20, cửa sổ nằm trọn trong 20% cuối), persistence đạt **MAE 3.16 ở "15 phút" — tốt hơn MAE 3.18 của T-GCN** trong bài báo; ở "60 phút" persistence có MAE 4.43 so với 4.60 của T-GCN (RMSE thì T-GCN tốt hơn: 5.13 so với 5.54 và 7.27 so với 8.46). Bài báo kết luận T-GCN vượt HA, ARIMA, SVR nhưng không cho thấy nó vượt được mức tối thiểu này theo MAE.

**P5. Kết luận không khớp với chính bảng số liệu.** Bài báo viết: *"the T-GCN model obtains the best prediction performance under all evaluation metrics for all prediction horizons"*. Nhưng trong Bảng 1, **MAE của GRU thấp hơn T-GCN ở cả 8 ô** (Los-loop: 3.06 < 3.18; 3.65 < 3.75; 4.09 < 4.12; 4.52 < 4.60 — SZ-taxi: 2.68 < 2.71; 2.70 < 2.75; 2.72 < 2.77; 2.74 < 2.79). T-GCN chỉ tốt hơn theo RMSE. Mục 5.3 giải thích cơ chế dẫn đến hiện tượng này.

**P6. "Không nhạy với tầm dự báo" là tính chất của dữ liệu, không phải của mô hình.** Trên SZ-taxi, RMSE của T-GCN chỉ tăng từ 3.92 (15 phút) lên 4.01 (60 phút). Nhưng chính trung bình trượt 3 giờ (HA cửa sổ) cũng gần như phẳng (4.16 → 4.31 trong thực nghiệm của chúng tôi): chuỗi SZ-taxi nhiễu và gần như dừng ở độ phân giải 15 phút, nên tầm dự báo ít quan trọng với *mọi* mô hình.

**P6b. Baseline HA trên SZ-taxi yếu bất thường.** Trên đúng các cửa sổ test của tác giả, trung bình 12 bước đầu vào (định nghĩa HA mà con số Los-loop của bài báo khớp: 7.48 / 3.97 so với 7.44 / 4.01) cho SZ-taxi **RMSE 4.17 / MAE 2.74**, trong khi bài báo ghi HA = 7.92 / 5.50. Với baseline đúng, khoảng cách "T-GCN tốt hơn HA 50.6%" biến mất: MAE của T-GCN (2.71) chỉ hơn trung bình trượt 3 giờ khoảng 1%, MAE của GRU (2.68) khoảng 2%.

**P7. Một lần chạy, không có độ biến thiên.** Bảng 1 không có độ lệch chuẩn hay kiểm định. Khi chênh lệch giữa GRU và T-GCN chỉ khoảng 1–3%, không thể biết nó có vượt nhiễu do khởi tạo ngẫu nhiên hay không (mục 8).

**P8. Dữ liệu.** Los-loop chỉ dài 7 ngày và chứa đoạn nội suy (gồm một lần mất dữ liệu toàn mạng); SZ-taxi có 24.8% giá trị 0 vốn là thiếu quan sát nhưng vẫn được dùng làm mục tiêu (mục 2.3).

### 4.1 Đo trực tiếp độ lạc quan của việc chọn epoch theo test

Mỗi run của chúng tôi ghi lại RMSE test ở *mọi* epoch (chỉ ghi, không dùng để chọn). So sánh RMSE test tại epoch có validation tốt nhất (cách đúng) với RMSE test nhỏ nhất qua mọi epoch (cách của tác giả):

ORACLE_TABLE

Chọn theo test làm sai số đẹp hơn trung bình ORACLE_MEAN% (tối đa ORACLE_MAX%) trên toàn bộ các run. Độ lạc quan **tăng theo mức dễ quá khớp**: 0.3–0.5% với GCN/GRU/T-GCN (đường học mượt), nhưng 2–5.6% với các mô hình nhiều tham số hoặc dừng sớm (tách self + RT, loss đơn vị gốc, Huber, đồ thị thích nghi) — đường sai số test của chúng dao động mạnh nên "chọn đỉnh đẹp nhất" được lợi nhiều hơn. Con số nghe nhỏ, nhưng **cùng cỡ với toàn bộ lợi thế mà bài báo báo cáo cho T-GCN so với GRU** trên Los-loop (RMSE "15 phút" 5.13 so với 5.22, tức 1.8%). Mục 9.2 tái lập nguyên vẹn giao thức của tác giả để so sánh trực tiếp.
