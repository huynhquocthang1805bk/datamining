# Ghi nhận kiểm thử khi đóng gói — 16/11/2026

## Đã thực hiện trong môi trường tạo ZIP
- `python -m unittest discover -s tests -v`: **4/4 test pass** (cửa sổ thời gian, chỉ số, kích thước tensor/gradient, ablation).
- Chạy GRU / T-GCN / T-GCN+ trên bộ dữ liệu **GIẢ LẬP** 520×10, 2 epochs, 3 train batches/epoch và 1 validation batch: cả 3 chạy xong, tạo checkpoint và các biểu đồ.
- `python evaluate.py` với checkpoint T-GCN: đọc và tính lại chỉ số thành công.
- `python compare.py` và `python make_report.py`: tạo bảng tổng hợp và mẫu báo cáo Word.
- Render Word ra ảnh để kiểm tra bố cục, font và không bị cắt nội dung.
- Dữ liệu thật từ GitHub tác giả **không thể tải trong môi trường đóng gói** do giới hạn truy cập mạng. `download_data.py` được soạn theo URL raw chính thức đã đối chiếu; cần chạy và kiểm thử lại trên thiết bị có internet.

## Điều kiện quan trọng
- Không coi kết quả 2-epoch/partial-batch trên demo là kết quả khoa học.
- Không báo cáo rằng biến thể cải tiến tốt hơn trên Los-loop hoặc SZ-taxi trước khi thực nghiệm thật.
- Không trích dẫn số MAE/RMSE demo trong phần kết quả báo cáo học thuật.
- Mỗi run thực có `metrics.json` và `config.json`; báo cáo tự sinh loại trừ các run có `--max-train-batches` hoặc `--max-val-batches`.
