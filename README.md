# T-GCN / T-GCN+ — Dự báo tốc độ giao thông

Dự án Python/PyTorch triển khai lại kiến trúc T-GCN (GCN + GRU) và một biến thể nghiên cứu **T-GCN+**. Có mã tải dữ liệu Los-loop/SZ-taxi, chạy thử offline, train/evaluate/so sánh, ảnh và tệp kết quả, cùng mẫu báo cáo Word.

> ĐỌC TRƯỚC: Đây là bản cài đặt **độc lập** theo ý tưởng T-GCN, không phải copy mã hoặc cam kết tái lập chỉ số của tác giả. Bản T-GCN+ mới là **giả thuyết cải tiến**, chưa được xác nhận bằng thực nghiệm thật. Bộ `data/demo` là dữ liệu GIẢ LẬP, chỉ dùng kiểm thử. Kết quả test trên demo không được trình bày như kết quả nghiên cứu.

## 1. Cài đặt

Yêu cầu Python 3.10 trở lên; khuyên dùng Python 3.10–3.12. Chạy bên trong thư mục chứa `train.py`.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS/Colab (Colab thường có torch):

```bash
pip install -r requirements.txt
```

Nếu dùng GPU NVIDIA, cần cài torch đúng bản CUDA theo https://pytorch.org/get-started/locally/ trước/hoặc thay cài mặc định.

## 2. Kiểm tra nhanh KHÔNG cần internet (synthetic)

Dữ liệu mô phỏng đã có tại `data/demo`. Có thể tạo lại bằng `python generate_demo.py`. Kiểm thử:

```bash
python -m unittest discover -s tests -v
python run_experiments.py --dataset demo --epochs 2 --batch-size 16 --max-train-batches 3 --max-val-batches 1 --device cpu
```

Kết quả nằm trong `results/demo/`. Tham số giới hạn batch **chỉ dành cho smoke test**, không dùng báo cáo học thuật.

## 3. Thử nghiệm THẬT trên Los-loop

Dữ liệu chưa được gói kèm ZIP do được tải từ repository công khai của tác giả. Cần internet khi chạy download:

```bash
python download_data.py --dataset losloop
python run_experiments.py --dataset losloop --epochs 80 --seeds 42 123 2026 --horizon 3 --seq-len 12 --device auto
python compare.py --dataset losloop
python make_report.py --dataset losloop
```

Muốn chạy nhanh thử trước:

```bash
python train.py --dataset losloop --model tgcn --epochs 5 --patience 3
python train.py --dataset losloop --model tgcn_plus --epochs 5 --patience 3
python compare.py --dataset losloop
```

Dữ liệu Los-loop có 207 cảm biến, 2016 mốc thời gian, mỗi mốc 5 phút (mặc định). `--horizon 3` dự báo trực tiếp 5, 10, 15 phút. Nếu muốn 15/30/45/60 phút, dùng `--horizon 12` và xem 3/6/9/12 trong `per_horizon` (mỗi model dùng cùng horizon).

Đổi sang SZ-taxi (156 đường, 2976 mốc, mỗi mốc 15 phút):

```bash
python download_data.py --dataset sztaxi
python run_experiments.py --dataset sztaxi --epochs 80 --seeds 42 123 2026 --horizon 4 --seq-len 12
python make_report.py --dataset sztaxi
```

Nếu mạng không cho tải tự động, mở link sau trên trình duyệt, tải CSV và đổi tên vào `data/losloop/`:

- `speed.csv`: https://raw.githubusercontent.com/lehaifeng/T-GCN/master/data/los_speed.csv
- `adjacency.csv`: https://raw.githubusercontent.com/lehaifeng/T-GCN/master/data/los_adj.csv

Tương tự `sz_speed.csv` và `sz_adj.csv` với thư mục `data/sztaxi/`.

## 4. Đánh giá checkpoint và trực quan hóa

```bash
python evaluate.py --checkpoint results/losloop/tgcn/seed_42/best.pt --plot-sensor 0
python evaluate.py --checkpoint results/losloop/tgcn_plus/seed_42/best.pt --plot-sensor 10
python compare.py --dataset losloop
python make_report.py --dataset losloop
```

Mỗi run lưu:

- `best.pt`: checkpoint sau epoch có val MSE thấp nhất
- `config.json`, `history.csv`, `training_summary.json`: cấu hình và lịch sử huấn luyện
- `metrics.json`: MAE/RMSE/MAPE/WAPE overall, theo mỗi horizon
- `test_predictions.npz`: `y_true`, `y_pred` có shape [mẫu, horizon, sensor], `time_indices`
- `learning_curve.png`: biến thiên validation MSE theo epoch
- `forecast_sensor.png`: biểu đồ đường thật theo thời gian thực, không nối các mẫu ngẫu nhiên
- `scatter.png`: phân tán dự đoán–thực tế

`results/<dataset>/comparison.csv`, `comparison_aggregated.csv`, `comparison.png` là bảng/ảnh so sánh. Tệp báo cáo tái tạo trong `reports/`.

## 5. Ablation — kiểm chứng đóng góp

Để đối chiếu tác dụng của từng cải tiến, chạy các lệnh sau với **cùng seq/horizon/split/seed**:

```bash
python train.py --dataset losloop --model tgcn_plus --seed 42 --result-dir results/losloop/ablations/full
python train.py --dataset losloop --model tgcn_plus --seed 42 --no-time-features --result-dir results/losloop/ablations/no_time
python train.py --dataset losloop --model tgcn_plus --seed 42 --no-adaptive-graph --result-dir results/losloop/ablations/no_graph
python train.py --dataset losloop --model tgcn_plus --seed 42 --no-residual --result-dir results/losloop/ablations/no_residual
```

Lưu ý: các run `--result-dir` ghi `model=tgcn_plus` trong CSV; có thể dùng cột `run` để phân biệt. So sánh hiệu quả dựa trên test RMSE/MAE; không gọi một cải tiến là hiệu quả nếu chưa đo trên cùng điều kiện.

## 6. Quy trình thực nghiệm để làm báo cáo

1. Chạy mọi mô hình trên **cùng** Los-loop/SZ-taxi, cùng cách tách train/validation/test theo thứ tự thời gian (70%/10%/20%).
2. Chuẩn hóa theo thống kê train mỗi cảm biến, không tính từ val/test.
3. Giữ seq_len, horizon, hidden_dim, epochs tối đa, patience, seed nhất quán khi hợp lý. Sự khác biệt loss của T-GCN+ phải được nêu rõ, có thể chạy ablation sau để tách tác dụng.
4. Sử dụng ít nhất 3 seed và so sánh trung bình ± độ lệch chuẩn; biểu đồ từ `results`.
5. Báo cáo riêng MAE, RMSE, MAPE (chỉ tốc độ >=5 đơn vị), WAPE; rõ đơn vị tốc độ theo nguồn.
6. Phân tích từng horizon và từng sensor; đề xuất cải tiến chỉ khi kết quả thực tế hỗ trợ.

## 7. Cấu trúc

```
TGCN_Traffic_Project/
  data/demo/                # Dữ liệu GIẢ LẬP được ghi rõ
  tgcn/                     # Data loader, model, metrics, training, plotting
  tests/                    # Unit tests
  reports/                  # Mẫu và báo cáo tạo tự động
  download_data.py          # Tải dataset THẬT từ T-GCN GitHub
  train.py / evaluate.py    # Huấn luyện và đánh giá
  run_experiments.py        # Baseline GRU, T-GCN, T-GCN+
  compare.py / make_report.py
  RUN_COLAB.ipynb           # Notebook chạy thử trên Colab
  CHANGES.md                # Giải thích cải tiến và khác biệt gốc
```

## 8. Nguồn, thừa nhận và giới hạn

- Bài báo: Zhao et al. (2019), *T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction*. IEEE TITS. https://arxiv.org/abs/1811.05320
- Repo tác giả, dữ liệu: https://github.com/lehaifeng/T-GCN ; https://github.com/lehaifeng/T-GCN/tree/master/data
- Dữ liệu không tải trước trong môi trường tạo ZIP, downloader hoạt động khi có internet. Không có cam kết GPU/performance khi chưa train trên dữ liệu thật.
- Timestamp của data CSV gốc không nằm trong từng hàng; mã dùng mốc bắt đầu và khoảng lấy mẫu từ tài liệu để sinh các feature thời gian, giả định chuỗi liên tục không mất bước.
- Đồ thị được đối xứng hóa và chuẩn hóa, phù hợp dữ liệu mẫu; nếu nghiên cứu hướng đường có chiều, cần chỉnh định nghĩa adjacency.

### Thử riêng tác dụng của Huber loss

```bash
python train.py --dataset losloop --model tgcn_plus --loss huber --result-dir results/losloop/ablations/huber_loss
```

**Chú ý:** Thí nghiệm chính để so sánh công bằng đều mặc định dùng `--loss mse`.

Có thể chạy **toàn bộ 5 thí nghiệm ablation** bằng một lệnh:

```bash
python run_ablations.py --dataset losloop --epochs 50 --seed 42
```
