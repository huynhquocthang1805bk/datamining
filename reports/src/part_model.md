## 6. Mô hình huấn luyện được là gì và cách đọc lên để dự báo

### 6.1 "Mô hình" thực chất là gì

Sau huấn luyện, mô hình là **một tệp checkpoint** (`best.pt`, 0.1–0.4 MB, phần lớn là ma trận kề 207 × 207 được lưu kèm; bản thân trọng số chỉ ≈ 60 KB) — chính là trạng thái ở epoch có MSE validation nhỏ nhất. Tệp này là một `dict` PyTorch gồm:

| Khóa | Nội dung |
|---|---|
| `state_dict` | các tensor trọng số đã học (bảng dưới) |
| `config` | toàn bộ siêu tham số của lần chạy: `model`, `seq_len`, `horizon`, `hidden_dim`, `residual`, `time_features`, `self_sep`, lr, seed... |
| `data_stats` | trung bình/độ lệch chuẩn của **train** cho từng cảm biến, ma trận Â đã chuẩn hóa, danh sách mã cảm biến, bước thời gian ⇒ checkpoint **tự đủ**, không cần đọc lại dữ liệu train |
| `best_epoch`, `best_val_mse_scaled` | epoch được chọn và sai số validation tương ứng |

Thẻ mô hình của T-GCN-RT (Los-loop, hidden = 64, L = 12, H = 12):

MODEL_CARD_TABLE

Điểm đáng chú ý: **mọi nút dùng chung một bộ trọng số** (W_g, W_c, W_out không phụ thuộc N). Vì vậy mô hình rất nhỏ (≈ 14 nghìn tham số cho 207 cảm biến); điều làm các nút khác nhau là (i) chuỗi đầu vào của chính nút và (ii) láng giềng của nó qua Â. Hệ quả thực tế: không thể dùng checkpoint Los-loop cho SZ-taxi (khác Â, khác thống kê chuẩn hóa), nhưng có thể thêm cảm biến mới nếu biết vị trí của nó trên đồ thị.

### 6.2 Quy trình dự báo khi "đọc" mô hình lên

Thư mục `models/` chứa: `losloop_gru.pt`, `losloop_tgcn.pt`, `losloop_tgcn_selfsep.pt`, `losloop_tgcn_res_time.pt` (mô hình so sánh — seed có loss validation nhỏ nhất) và **`losloop_gru_res_time_seed{42,123,2026}.pt` — mô hình cuối cùng, dùng chung như một ensemble**. `models/index.json` ghi nguồn gốc và sai số test của từng tệp.

```
1. Đọc checkpoint         ck = torch.load('models/losloop_gru_res_time_seed42.pt')
2. Dựng lại kiến trúc     model = build_model(Namespace(**ck['config']), Â)   # từ data_stats
3. Nạp trọng số           model.load_state_dict(ck['state_dict']); model.eval()
4. Lấy 12 mốc gần nhất    X_raw: [12, 207]  (tốc độ thô, mph)
5. Chuẩn hóa              X = (X_raw − mean_train) / std_train
6. Đặc trưng thời gian    sin/cos(giờ), sin/cos(thứ) của 12 mốc đó
7. Chạy mô hình           ŷ = model(X, time) → [12, 207]   (12 bước × 207 cảm biến)
                          (residual: ŷ = x_t + Δ̂ được cộng sẵn bên trong mô hình)
8. Đổi về đơn vị gốc      Ŷ = ŷ · std_train + mean_train
9. Ensemble + hậu xử lý   trung bình Ŷ của 3 checkpoint, cắt về Ŷ ≥ 0
```

Toàn bộ quy trình được đóng gói trong `predict.py` (lớp `Forecaster`; truyền nhiều `--checkpoint` để ensemble). Ví dụ dự báo lúc 07:30 sáng thứ Tư 07/03/2012 (giờ cao điểm, nằm trong tập test) bằng mô hình cuối và đối chiếu với thực tế:

```bash
python predict.py --checkpoint models/losloop_gru_res_time_seed*.pt --input data/losloop/speed.csv --end-time "2012-03-07 07:30" --compare --sensors 4
```

PREDICT_EXAMPLE

Ba cảm biến thông thoáng (773869, 767541, 717447) được dự báo sát, sai số vài mph. Cảm biến 767542 đang kẹt (~18–20 mph): mô hình dự báo tốc độ **hồi phục dần lên ~38 mph sau 1 giờ** trong khi thực tế vẫn kẹt — đúng thiên lệch "lạc quan khi kẹt xe" đã đo ở mục 8.5. Đây là cách đọc kết quả đúng: con số dự báo là kỳ vọng có điều kiện, nên trong tình huống bất định (kẹt sẽ tan hay không) nó nằm giữa hai kịch bản.

Đầu ra có thể ghi ra CSV (`--output forecast.csv`, H dòng × N cột). Muốn dùng dữ liệu mới: chỉ cần CSV cùng thứ tự cột cảm biến và thời điểm bắt đầu (`--start-time`).
