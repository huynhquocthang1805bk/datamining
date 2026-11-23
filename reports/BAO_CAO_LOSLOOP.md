# DỰ BÁO TỐC ĐỘ GIAO THÔNG BẰNG T-GCN VÀ T-GCN CẢI TIẾN

## 1. Giới thiệu
Dự báo tốc độ giao thông nhiều vị trí là bài toán chuỗi thời gian có phụ thuộc không gian.
T-GCN kết hợp Graph Convolutional Network (GCN) và Gated Recurrent Unit (GRU),
nhằm khai thác đồng thời kết nối giữa các cảm biến và biến thiên theo thời gian [1].
Mục tiêu đồ án: cài đặt lại mô hình, so sánh baseline GRU với T-GCN,
thử một biến thể T-GCN+ và kiểm tra từng thành phần cải tiến.

## 2. Dữ liệu và tiền xử lý
Bộ dữ liệu: **Los-loop** (207 cảm biến, 2016 mốc thời gian, 5 phút/mốc).
Chỉ dùng dữ liệu tốc độ và ma trận kề; không tự nhận có nhãn kẹt xe, thời tiết hoặc lưu lượng.
Chia theo thời gian: 70% train, 10% validation và 20% test (không trộn ngẫu nhiên).
Dùng z-score theo cảm biến với trung bình và độ lệch chuẩn **chỉ tính từ train**.
Cửa sổ vào mặc định dài 12 bước; dự báo trực tiếp 3 bước.
Mỗi bước là 5 phút. Mô hình có thể dùng bước lịch tuần hoàn nếu biết ngày giờ bắt đầu.

## 3. Phương pháp
**T-GCN cơ sở:** Với A là ma trận kề, đặt A_tilde = A + I (chỉ thêm self-loop nếu chưa có),
A_bar = D^(-1/2) A_tilde D^(-1/2). Mỗi cổng GRU dùng phép biến đổi
GCN(X) = A_bar X W + b. Với trạng thái H tại thời điểm t-1:

z_t = sigmoid(GCN([x_t, H_(t-1)]));
r_t = sigmoid(GCN([x_t, H_(t-1)]));
H~_t = tanh(GCN([x_t, r_t * H_(t-1)]));
H_t = z_t * H_(t-1) + (1-z_t) * H~_t.

Một lớp tuyến tính biến trạng thái cuối thành dự báo đa bước tại mỗi nút.
Baseline GRU chỉ học diễn biến chuỗi tại từng nút mà không đưa ma trận kề vào mô hình.

**T-GCN+ (giả thuyết cải tiến):**

1. Học đồ thị mềm A_learn = softmax(ReLU(E E^T)), rồi phối hợp với A_bar qua alpha học được.
2. Bổ sung sin/cos của giờ trong ngày và thứ trong tuần.
3. Head dự báo phần thay đổi, cộng với tốc độ gần nhất (residual forecasting).
4. Dùng dropout và early stopping bằng validation MSE; thử Huber (Smooth L1) riêng nếu cần.

Cần thực nghiệm ablation để quy kết tác dụng từng thành phần; chưa thể khẳng định cải tiến tốt hơn.

## 4. Thiết kế thí nghiệm
Huấn luyện tất cả mô hình trên cùng bộ dữ liệu, cùng horizon, cùng train/val/test và seed.
Chọn checkpoint theo MSE nhỏ nhất trên validation; chỉ dùng test để báo cáo sau cùng.
Đánh giá MAE, RMSE, WAPE, MAPE (với |giá trị thật| >= 5 đơn vị; phải ghi tỷ lệ bao phủ).
Ưu tiên tối thiểu ba seed và báo cáo trung bình ± độ lệch chuẩn.
Tốc độ dùng đơn vị do nguồn dữ liệu cung cấp, cần đối chiếu metadata khi công bố.

## 5. Kết quả thực nghiệm

*Chưa có kết quả huấn luyện đầy đủ cho dữ liệu này. Bảng MAE/RMSE/MAPE sẽ tự cập nhật sau khi chạy `python make_report.py --dataset losloop`.*

## 6. Thảo luận và giới hạn
- Phân tích lỗi theo chân trời dự báo và từng vị trí; trình bày ví dụ biểu đồ chuỗi thời gian.
- So sánh GRU với T-GCN để đánh giá vai trò thông tin đồ thị.
- So sánh T-GCN với T-GCN+ và từng phép bỏ thành phần để đánh giá đóng góp thực sự.
- Không suy diễn rằng MAE/RMSE nhỏ đồng nghĩa phát hiện tắc nghẽn chính xác: đây là bài toán **dự báo tốc độ**, chưa có phân loại tắc nghẽn.
- Mô hình gốc ở đây là cài đặt độc lập, khác thiết lập gốc về chuẩn hóa, chia tập, thời gian dự báo và framework; vì vậy không trực tiếp so chỉ số với bài báo.
- Đặc trưng giờ/ngày được xây dựng từ mốc thời gian và giả định lấy mẫu liên tục; nếu có thiếu thời điểm phải chỉnh lại.
- Graph được đối xứng hóa; nghiên cứu đồ thị có hướng cần kiểm tra định nghĩa cạnh thực tế.

## 7. Kết luận
Đánh giá đóng góp chỉ dựa trên kết quả thực nghiệm của dữ liệu thật, có kiểm soát seed và ablation.
Nếu T-GCN+ không cải thiện hoặc độ chênh lệch thấp hơn độ biến động giữa seed, cần báo cáo trung thực.

## Tài liệu tham khảo
[1] Zhao, L. et al. (2019). T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction.
IEEE Transactions on Intelligent Transportation Systems. https://arxiv.org/abs/1811.05320

[2] Mã nguồn và dữ liệu của nhóm tác giả: https://github.com/lehaifeng/T-GCN
