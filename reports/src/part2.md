## 3. Phương pháp

### 3.1 Các mô hình

Ký hiệu: x_t ∈ R^N là tốc độ (đã chuẩn hóa z-score theo từng cảm biến, thống kê **chỉ lấy từ train**) tại bước t; A là ma trận kề; Â = D̃^(-1/2)(A + I)D̃^(-1/2) là ma trận kề chuẩn hóa đối xứng (đúng như mã tác giả: A + I với A gốc — Los-loop đã có đường chéo 1 nên self-loop thực tế là 2; SZ-taxi có hướng được giữ nguyên hướng, Â = D^(-1/2)(A+I)ᵀD^(-1/2)).

| Mô hình | Mô tả | Thông tin dùng |
|---|---|---|
| Naive (persistence) | ŷ_{t+k} = x_t | chỉ giá trị cuối |
| HA (trung bình cửa sổ) | ŷ_{t+k} = trung bình L bước vào — đúng như "HA" của tác giả (giống nhau ở mọi tầm) | thời gian |
| HA theo giờ | trung bình cùng khung giờ trên các ngày train | lịch |
| Hồi quy tuyến tính | ridge trên L độ trễ, dùng chung cho mọi nút (tương đương SVR tuyến tính của bài báo) | thời gian (tuyến tính) |
| GCN | 2 lớp GCN, L độ trễ là đặc trưng của nút, không có trạng thái hồi quy | không gian |
| GRU | một GRU dùng chung trọng số cho mọi nút, không trao đổi giữa các nút | thời gian |
| **T-GCN** | GRU mà mọi phép tuyến tính trong cổng được thay bằng GCN (công thức dưới) | không gian + thời gian |

Ô T-GCN (cùng cấu trúc với mã TensorFlow của tác giả: một lớp GCN trong mỗi cổng, bias cổng khởi tạo 1.0):

```
u_t = σ( Â [x_t , h_{t-1}] W_u + b_u )          cổng cập nhật (giữ bộ nhớ bao nhiêu)
r_t = σ( Â [x_t , h_{t-1}] W_r + b_r )          cổng reset
c_t = tanh( Â [x_t , r_t ⊙ h_{t-1}] W_c + b_c )  trạng thái ứng viên
h_t = u_t ⊙ h_{t-1} + (1 − u_t) ⊙ c_t
ŷ_{t+1..t+H} = h_L W_out + b_out                 (một đầu ra tuyến tính cho H bước, dự báo trực tiếp)
```

GRU baseline chính là ô trên với Â = I. Vì vậy so sánh GRU ↔ T-GCN cô lập đúng một yếu tố: **có hay không trộn thông tin láng giềng**.

### 3.2 Các cải tiến được thử (giai đoạn 3)

| Ký hiệu | Thay đổi | Lý do (giả thuyết) |
|---|---|---|
| `+time` | nối sin/cos(giờ trong ngày), sin/cos(thứ trong tuần) vào đầu vào mỗi nút | giờ cao điểm lặp lại hằng ngày; GRU 12 bước không "biết" bây giờ là mấy giờ |
| `+res` (residual) | ŷ_{t+k} = x_t + Δ̂_k: mạng chỉ học **phần thay đổi** so với giá trị cuối | persistence đã rất mạnh ở tầm ngắn; Â làm loãng tín hiệu của chính nút (mục 5.3) |
| **T-GCN-RT** | T-GCN + time + res | kết hợp hai ý trên |
| `selfsep` | GCN dạng [x, Âx]W: trọng số riêng cho bản thân và láng giềng | giữ 100% tín hiệu của chính nút (như số hạng k = 0 của DCRNN) |
| T-GCN+ | RT + đồ thị thích nghi học từ embedding nút | bắt quan hệ không có trong đồ thị vật lý |
| Huber | hàm mất mát Smooth-L1 thay MSE | bền với ngoại lai, nhắm MAE |
| cửa sổ L, hidden | L ∈ {6, 12, 24}, hidden ∈ {32, 64, 100} | theo yêu cầu đề tài |
| Ensemble | trung bình dự báo của 3 seed | giảm phương sai, tăng độ tin cậy |

### 3.3 Thiết kế thực nghiệm (giao thức "nghiêm ngặt")

- **Chia theo thời gian 70/10/20** (train/validation/test), không xáo trộn; chuẩn hóa bằng thống kê của train; ngữ cảnh đầu vào của mẫu test được phép lấy từ quá khứ (val), nhưng **mục tiêu** luôn nằm trong 20% cuối.
- Đầu vào L = 12 bước (Los-loop: 60 phút; SZ-taxi: 3 giờ); dự báo trực tiếp H = 12 bước (Los-loop, 5…60 phút) hoặc H = 4 (SZ-taxi, 15…60 phút). Một mô hình cho mọi tầm.
- Huấn luyện: AdamW, lr 2·10⁻³ giảm dần kiểu cosine đến 10⁻⁴, weight decay 10⁻⁴, batch 32, tối đa 120 epoch, **dừng sớm theo MSE trên validation** (patience 20), cắt gradient 5. Hàm mất mát MSE cho mọi mô hình (Huber chỉ là biến thể).
- **3 seed** (42, 123, 2026) cho các mô hình chính và T-GCN-RT; báo cáo trung bình ± độ lệch chuẩn. Biến thể phụ và tinh chỉnh: 1 seed (42).
- Đánh giá: RMSE, MAE, MAPE (chỉ |y| ≥ 5), WAPE và ba chỉ số của bài báo (Accuracy, R², Var). Sai số được tính **tại đúng bước k** ("per-step"); để so với bài báo, chúng tôi tính thêm kiểu của tác giả: gộp mọi bước 1..k ("cumulative").
- Để đo độ lạc quan của giao thức tác giả, mỗi run ghi lại RMSE test ở mọi epoch (**chỉ ghi, không dùng để chọn**).
- Phần cứng: CPU 4 nhân, không GPU (≈ 15–25 giây/epoch cho T-GCN, 4 run song song).
