# Ghi nhận kiểm thử — 23/11/2026

- `python -m unittest discover -s tests -v`: **8/8 test pass** (rò rỉ nhãn giữa các split, kích thước/gradient của mọi mô hình và tùy chọn, residual với đầu ra 0 = persistence, sparse = dense, giao thức paper, định nghĩa Accuracy/R²/Var và cumulative, mask số 0).
- Dữ liệu **thật** Los-loop và SZ-taxi tải từ repo tác giả; mốc thời gian dựng lại được kiểm chứng bằng hồ sơ giờ cao điểm (Hình 1 của báo cáo).
- ≈ 60 run huấn luyện đầy đủ (CPU 4 nhân), 3 seed cho mọi mô hình chính; mọi bảng trong báo cáo được sinh từ `metrics.json` bởi `tools/build_report.py`, không chép tay.
- Kiểm tra hợp lệ của phân tích: xáo trộn láng giềng có tác động đúng bằng 0 với GRU (không dùng đồ thị) và với nút xa; saliency của GCN 2 lớp bằng 0 từ hop thứ 3; GRU tái lập Bảng 1 (giao thức tác giả: RMSE 5.233, Accuracy 0.9109, R² 0.8574 so với 5.218 / 0.9109 / 0.8576).
- Giới hạn: Los-loop chỉ 7 ngày; ≤ 120 epoch (giao thức chính), 300/150 epoch (tái lập giao thức tác giả) so với 3000 của tác giả.
