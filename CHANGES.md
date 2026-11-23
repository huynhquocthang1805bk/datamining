# Thay đổi so với T-GCN gốc

## Mô hình giữ nguyên ý tưởng của Zhao et al.
- Dùng tích chập đồ thị với ma trận kề chuẩn hóa trong các cổng cập nhật/đặt lại của GRU.
- Dự báo đa bước trực tiếp bằng một head tuyến tính cho từng nút.
- Không sử dụng nhãn kiểm tra khi học hoặc chọn checkpoint.

## T-GCN+ (giả thuyết nghiên cứu, chưa khẳng định cải thiện)
1. **Adaptive graph mixing**: học ma trận tương đồng từ embedding của sensor và trộn với graph có sẵn theo alpha học được. Mục tiêu: bắt quan hệ chưa thể hiện trong graph vật lý.
2. **Cyclical time embeddings**: sin/cos giờ trong ngày và ngày trong tuần đưa trực tiếp vào các cổng. Mục tiêu: ghi nhận tính chu kỳ.
3. **Residual prediction**: cộng giá trị cuối của cửa sổ đầu vào với thay đổi dự báo. Mục tiêu: học xu hướng biến động thay vì giá trị tuyệt đối.
4. **Dropout và early stopping**; tất cả mô hình mặc định **MSE** để so sánh công bằng. **Huber (Smooth L1)** là thử nghiệm tùy chọn bằng `--loss huber`, không nằm trong cấu hình chính.

Tất cả là **thay đổi đề xuất**, phải thực nghiệm và ablation mới có cơ sở khẳng định chúng hiệu quả.

## Khác biệt quan trọng cần nêu trong báo cáo
- Đây là **tái triển khai độc lập**, không phải kết quả tái lập chính xác mã TensorFlow gốc.
- Chuẩn hóa per-sensor z-score tính trên train, thay vì min-max trong một số mã mẫu.
- Chia theo thời gian 70/10/20 (train/val/test), target test chỉ thuộc 20% cuối. Các số đo KHÔNG trực tiếp so sánh với bài báo dùng protocol khác.
- Đánh giá đa bước trực tiếp, horizon mặc định 3; Los-loop 5 phút/bước = 15 phút.
- MAPE chỉ tính với tốc độ thực có |y| >= 5 để tránh mẫu số gần 0; báo cáo kèm coverage.
- Dataset `demo` hoàn toàn tổng hợp, không dùng làm bằng chứng khoa học.
