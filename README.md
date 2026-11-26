# GRU / T-GCN — Dự báo tốc độ giao thông: tái lập, phản biện và cải tiến

Cài đặt PyTorch độc lập của GRU, GCN, T-GCN (Zhao et al., 2019) và các cải tiến, chạy trên dữ liệu thật **Los-loop** (207 cảm biến, 5 phút) và **SZ-taxi** (156 tuyến, 15 phút) theo ba giai đoạn của đề tài:

1. GRU — kết quả ban đầu, kiểm tra dữ liệu, biểu đồ chuỗi thời gian;
2. T-GCN — cùng dữ liệu, cùng cách chia train/validation/test;
3. So sánh sai số, phân tích giờ cao điểm, thử cải tiến (cửa sổ đầu vào, hidden units, đặc trưng thời gian, residual, cách dùng đồ thị).

📄 **Báo cáo đầy đủ:** [`reports/BAO_CAO_TGCN.md`](reports/BAO_CAO_TGCN.md) (bản Word: `reports/BAO_CAO_TGCN.docx`) — ý tưởng → dẫn dắt → phản biện → minh chứng; ảnh hưởng của thuộc tính; mô hình huấn luyện được và cách nạp để dự báo; độ tin cậy; giải thích cơ chế; so sánh với bài báo.

## Kết quả chính so với baseline gốc T-GCN (Los-loop, test 20% cuối, 3 seed, RMSE / MAE trung bình 12 tầm 5…60 phút)

**Số in đậm = tốt hơn T-GCN gốc** (dòng *baseline*, tái lập cùng giao thức). Bảng đầy đủ — cả so với con số T-GCN trong Bảng 1 của bài báo, cả SZ-taxi — ở đầu báo cáo (mục "Bảng tổng hợp").

| Mô hình | RMSE | MAE | Δ RMSE / MAE so với T-GCN gốc | Ghi chú |
|---|---|---|---|---|
| Persistence (giữ giá trị cuối) | 8.42 | 4.41 | +12.0% / +0.9% | baseline bài báo không có |
| GRU | **7.50** ± 0.01 | **4.18** ± 0.02 | −0.3% / −4.5% | tái lập đúng bài báo ("15 phút": 5.217 so với 5.218) |
| T-GCN gốc (Â đúng như mã tác giả) — *baseline* | 7.52 ± 0.03 | 4.37 ± 0.04 | — | không hơn GRU (p = 0.76) |
| T-GCN tách self/láng giềng `[x, Âx]W` | **7.30** ± 0.06 | **4.12** ± 0.02 | −2.9% / −5.7% | hơn GRU có ý nghĩa (p = 0.01) |
| STID / DCRNN-lite (kiến trúc 2018–2022) | **7.43** / 7.56 | **4.16** / 4.39 | −1.2% / +0.6% (RMSE) | quá khớp trên 5 ngày dữ liệu |
| STAEformer cỡ đầy đủ, 1.26 triệu tham số (GPU) | **7.22** ± 0.43 | **4.05** ± 0.17 | −3.9% / −7.3% | quá khớp, rất không ổn định giữa các seed |
| GRU-RT (residual + giờ/thứ), ensemble 3 seed | **6.91** | **4.01** | −8.1% / −8.2% | mô hình tốt nhất phần 1 |
| T-GCN-RTSP (+ hồ sơ theo giờ của cảm biến) | **6.69** ± 0.03 | **4.10** ± 0.09 | −11.1% / −6.2% | tốt nhất trong họ T-GCN |
| Graph WaveNet + residual (2019) — CPU / GPU | **6.59** ± 0.12 / **6.57** ± 0.05 | **3.67** / **3.74** | −12.3% / −15.9% (CPU) | mạng nơ-ron tốt nhất; GPU tái lập CPU |
| LightGBM + đặc trưng không gian–thời gian | **6.47** ± <0.01 | **3.60** | −14.0% / −17.7% | mô hình đơn tốt nhất |
| Trộn LightGBM + Graph WaveNet-res theo tầm (trọng số fit trên validation) | **6.19** | **3.49** | **−17.6% / −20.1%** | **tốt nhất**; vượt T-GCN của bài báo ở 16/16 ô Bảng 1 |

SZ-taxi (mục tiêu ≠ 0; T-GCN gốc 4.69 / 3.35 là baseline): GRU **4.62** / **3.27**, LightGBM **4.49** / **3.17**, STID **4.44** / **3.14**, Graph WaveNet (GPU) **4.43** / **3.13**, trộn STID + LightGBM **4.40** / **3.11** (−6.2% / −7.2%).

Phát hiện chính:
- Phép chuẩn hóa ma trận kề của T-GCN chỉ giữ ~30% tín hiệu của chính nút ⇒ dự báo tầm ngắn kém cả persistence.
- Bài báo chọn epoch theo tập test, gộp sai số các tầm, thiếu baseline persistence, và MAE của GRU tốt hơn T-GCN ở cả 8 ô của Bảng 1.
- Đặc trưng quan trọng nhất là **hồ sơ tốc độ theo khung giờ của từng cảm biến** (50–65% tầm quan trọng trong LightGBM). Nhờ nó, lần đầu có mô hình thắng persistence trong giờ cao điểm.
- Kiến trúc hiện đại phụ thuộc dữ liệu: trên Los-loop (5 ngày) chỉ Graph WaveNet + residual phát huy, trên SZ-taxi (22 ngày) STID và Graph WaveNet dẫn đầu. Chạy cỡ đầy đủ trên GPU (42 run, mục 11.8 của báo cáo) không cải thiện: giới hạn là dữ liệu, không phải sức tính — bước tiếp theo là METR-LA / PEMS-BAY ([`LOCAL_RUN.md`](LOCAL_RUN.md)).

## 1. Cài đặt

Python 3.10+.

```bash
pip install -r requirements.txt
```

(Trên Windows: tạo venv bằng `py -m venv .venv` rồi `.venv\Scripts\Activate.ps1`. GPU: cài torch đúng bản CUDA theo pytorch.org. Toàn bộ thí nghiệm trong báo cáo chạy trên CPU 4 nhân.)

## 2. Dữ liệu

```bash
python download_data.py --dataset losloop
python download_data.py --dataset sztaxi
```

Tải từ repo của tác giả (`lehaifeng/T-GCN/data`). Nếu không tải tự động được: tải `los_speed.csv`, `los_adj.csv` (hoặc `sz_*`) và đặt thành `data/<dataset>/speed.csv`, `data/<dataset>/adjacency.csv`. `data/demo/` là dữ liệu **giả lập** chỉ để kiểm thử.

## 3. Kiểm thử nhanh

```bash
python -m unittest discover -s tests -v
python train.py --dataset demo --model tgcn --epochs 2 --max-train-batches 3 --max-val-batches 1 --device cpu
```

## 4. Chạy lại toàn bộ thí nghiệm

```bash
python baselines.py --dataset losloop                 # persistence, HA, HA theo giờ, hồi quy tuyến tính
python baselines.py --dataset sztaxi
python run_suite.py --spec losloop:main losloop:improve losloop:variants losloop:tuning --jobs 4
python run_suite.py --spec losloop:paper sztaxi:paper sztaxi:main sztaxi:improve --jobs 4
```

`run_suite.py` chạy song song và tiếp tục được khi bị ngắt (bỏ qua run đã có `metrics.json`). Nhóm thí nghiệm:

| Nhóm | Nội dung |
|---|---|
| `main` | GRU, GCN, T-GCN × 3 seed (giai đoạn 1–2) |
| `improve` | GRU-RT, T-GCN-RT (+ các biến thể tốt nhất) × 3 seed; SZ-taxi: loss bỏ qua số 0 |
| `variants` | từng thành phần một: +residual, +thời gian, tách self/láng giềng, Â = I, đồ thị thích nghi, Huber, loss đơn vị gốc, self-loop = 1 |
| `tuning` | cửa sổ L ∈ {6, 12, 24}, hidden ∈ {32, 64, 100} |
| `paper` | tái lập đúng giao thức tác giả (`--protocol paper`: 80/20, chuẩn hóa max, chọn epoch theo test) |

Một run lẻ, ví dụ T-GCN-RT tách self/láng giềng:

```bash
python train.py --dataset losloop --model tgcn --self-sep --residual --time-features --horizon 12 --seed 42
```

Tùy chọn của `train.py`: `--model {gru,gcn,tgcn,tgcn_plus}`, `--residual`, `--time-features`, `--self-sep`, `--no-graph`, `--adaptive-graph`, `--loss {mse,huber,mae}`, `--raw-loss`, `--mask-zero`, `--normalization {zscore,max}`, `--protocol {strict,paper}`, `--track-test` (chỉ ghi RMSE test theo epoch để đo độ lạc quan, không dùng để chọn).

### Cải tiến nâng cao (mục 11 của báo cáo)

```bash
python run_suite.py --spec losloop:modern --jobs 4          # STID, DCRNN-lite, GRU-RTP, T-GCN-RTSP, Graph WaveNet (gọn)
python gbdt.py --dataset losloop --features 2 --num-leaves 127 --tag lgbm_best --seeds 42 123 2026
python analysis/blend.py --dataset losloop --members lgbm_best gwnet_res gru_res_time tgcn_rtsp
python analysis/modern_models.py --dataset losloop
```

Các cấu hình nặng (STAEformer, Graph WaveNet cỡ đầy đủ, METR-LA/PEMS-BAY): xem [`LOCAL_RUN.md`](LOCAL_RUN.md). Kết quả GPU đã chạy được lưu trong `reports/external/`; kiểm tra và tổng hợp:

```bash
python analysis/gpu_results.py        # kiểm tra số liệu GPU, suy ra chỉ số kiểu bài báo
python analysis/baseline_compare.py   # bảng mọi mô hình so với baseline gốc T-GCN (in đậm số tốt hơn)
```

## 5. Phân tích và báo cáo

```bash
python analysis/eda.py                                   # giai đoạn 1: kiểm tra dữ liệu, biểu đồ
python analysis/compare_models.py --dataset losloop      # bảng chính, ensemble, so với bài báo, độ lạc quan
python analysis/peak_hours.py --dataset losloop          # giờ cao điểm, kẹt xe, biến động mạnh
python analysis/tuning_ablation.py --dataset losloop     # tinh chỉnh + ablation
python analysis/feature_importance.py --dataset losloop  # ảnh hưởng thuộc tính, saliency
python analysis/explain.py --dataset losloop             # skill, cổng GRU, Δ dự báo, thẻ mô hình
python analysis/reliability.py --dataset losloop         # bootstrap, Diebold–Mariano, nhiễu, conformal
python analysis/paper_protocol.py                        # tái lập giao thức tác giả
python tools/export_models.py                            # models/*.pt (seed chọn theo validation)
python tools/build_report.py                             # reports/BAO_CAO_TGCN.md (bảng điền từ CSV)
python tools/md2docx.py reports/BAO_CAO_TGCN.md reports/BAO_CAO_TGCN.docx
```

Hình ở `reports/figures/`, bảng số liệu ở `reports/tables/`. Mỗi run trong `results/<dataset>/...` có `config.json`, `history.csv`, `metrics.json` (per-step, cumulative kiểu bài báo, Accuracy/R²/Var, bản bỏ số 0 cho SZ-taxi), `test_predictions.npz`, `best.pt`, `learning_curve.png`.

## 6. Dùng mô hình đã huấn luyện để dự báo

```bash
python predict.py --checkpoint models/losloop_gru_res_time_seed*.pt --input data/losloop/speed.csv --end-time "2012-03-07 07:30" --compare
```

Checkpoint tự chứa trọng số, cấu hình và thống kê chuẩn hóa của train; truyền nhiều checkpoint để lấy trung bình (ensemble); dự báo được cắt về ≥ 0. `models/index.json` mô tả từng tệp.

Mô hình LightGBM (thư mục gồm 12 mô hình cây — một cho mỗi tầm — và `stats.npz`):

```bash
python predict.py --checkpoint models/losloop_lgbm --input data/losloop/speed.csv --end-time "2012-03-07 07:30" --compare
```

## 7. Cấu trúc

```
tgcn/            data.py · models.py (GRU, GCN, T-GCN) · models_modern.py (STID, Graph WaveNet, DCRNN-lite, STAEformer)
                 training.py · metrics.py · plotstyle.py
baselines.py     gbdt.py (LightGBM) · train.py · evaluate.py · predict.py · run_suite.py
analysis/        eda · compare_models · peak_hours · tuning_ablation · feature_importance · explain · reliability
                 paper_protocol · modern_models · blend
tools/           export_models · build_report · md2docx · prepare_large_dataset (METR-LA / PEMS-BAY)
models/          checkpoint cuối cùng
reports/         BAO_CAO_TGCN.md/.docx, figures/, tables/
tests/           8 unit test
```

`run_experiments.py`, `compare.py`, `make_report.py`, `run_ablations.py` là công cụ của phiên bản đầu, vẫn chạy được nhưng đã được thay bằng `run_suite.py` và `analysis/`.

## 8. Nguồn và giới hạn

- Zhao et al. (2019), *T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction*, IEEE T-ITS — arXiv:1811.05320; mã và dữ liệu: github.com/lehaifeng/T-GCN.
- Li et al. (2018), *Diffusion Convolutional Recurrent Neural Network* — gợi ý cho cách tích chập đồ thị giữ tín hiệu của chính nút.
- Cài đặt độc lập, không phải mã gốc. Los-loop chỉ 7 ngày (test ≈ 1.4 ngày); huấn luyện ≤ 120 epoch trên CPU (tác giả: 3000). Dữ liệu không có cột thời gian: mốc được dựng từ ngày bắt đầu trong bài báo (đã kiểm chứng bằng hồ sơ giờ cao điểm).
