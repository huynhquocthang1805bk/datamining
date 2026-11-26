## 11. Cải tiến nâng cao: kiến thức và công nghệ mới

### 11.1 Ý tưởng — bốn hướng thử

Sau mục 5, mô hình tốt nhất là ensemble GRU-RT (RMSE 6.91 / MAE 4.01) và điểm yếu chung là giờ cao điểm. Chúng tôi thử thêm bốn hướng, **cùng giao thức nghiêm ngặt** (70/10/20, thống kê chỉ từ train, chọn theo validation, 3 seed):

| Hướng | Mô hình | Ý tưởng chính |
|---|---|---|
| A. Kiến trúc hiện đại (2018–2023) | **STID** (Shao et al., 2022) | MLP + embedding "danh tính" của cảm biến, giờ trong ngày, thứ — không đồ thị, không hồi quy |
| | **DCRNN-lite** (Li et al., 2018 — bài thứ 2 trong Papers.xlsx) | GRU với diffusion convolution hai chiều K = 2, *giữ trọng số riêng cho chính nút* (k = 0) |
| | **Graph WaveNet** (Wu et al., 2019) | TCN giãn nở có cổng + GCN khuếch tán + đồ thị thích nghi |
| | **STAEformer** (Liu et al., 2023) | Transformer theo thời gian và theo nút + embedding thích nghi không gian–thời gian |
| B. Khai phá dữ liệu cổ điển | **LightGBM** + đặc trưng không gian–thời gian | ~50 đặc trưng thiết kế tay: độ trễ, thống kê trượt, trung bình láng giềng (Âx), lịch, **hồ sơ tốc độ theo khung giờ của từng cảm biến** (tính từ train); một mô hình cây tăng cường cho mỗi tầm, dự báo phần thay đổi |
| C. Chuyển kiến thức B → học sâu | **GRU-RTP**, **T-GCN-RTSP** | thêm hồ sơ theo giờ của cảm biến làm đầu vào và "thay đổi hồ sơ dự kiến tới t+k" vào đầu ra |
| D. Kết hợp | **Trộn theo từng tầm** | trọng số không âm (tổng 1) cho từng tầm dự báo, **tối ưu trên validation**, rồi mới áp dụng lên test |

Trên CPU của chúng tôi, Graph WaveNet bản gọn chạy 1 seed không residual và 3 seed có residual (≈ 30–50 phút/run). STAEformer (83–390 s/epoch trên CPU) và các cỡ đầy đủ theo bài gốc được **chạy trên GPU máy của bạn**: 7 cấu hình × 3 seed × 2 bộ dữ liệu = 42 run, cùng giao thức (`run_suite.py --groups modern_gpu`, `LOCAL_RUN.md`). Các dòng "(GPU)" trong các bảng dưới là các run đó; mục 11.8 kiểm tra số liệu và phân tích riêng.

### 11.2 Kết quả trên Los-loop

MODERN_TABLE

![Hình 20. Xếp hạng mọi mô hình trên cùng tập test Los-loop](figures/modern_ranking_losloop.png)

![Hình 21. Sai số theo tầm dự báo của các mô hình mới](figures/modern_horizon_losloop.png)

Kiểm định cặp so với mô hình tốt nhất của phần trước (ensemble GRU-RT), bootstrap khối 1 giờ + Diebold–Mariano:

MODERN_TESTS

Năm kết luận:

1. **Mô hình tốt nhất mới là trộn theo tầm LightGBM + Graph WaveNet-residual (+ GRU-RT): RMSE 6.19 / MAE 3.49** — giảm 10% RMSE so với ensemble GRU-RT (p = 0.0002), **18% RMSE và 20% MAE so với T-GCN gốc**, 17% / 16% so với GRU.
2. **LightGBM với đặc trưng thiết kế tay là mô hình đơn tốt nhất (6.47 / 3.60)**, cực kỳ ổn định giữa các seed (± 0.001), hơn ensemble GRU-RT có ý nghĩa (p = 0.03).
3. **Kiến trúc hiện đại: kết quả phân hóa.** **Graph WaveNet + residual (6.59 ± 0.12 / 3.67)** là mô hình đơn tốt thứ hai và là mạng nơ-ron tốt nhất (p = 0.003 so với ensemble GRU-RT) — nhưng *không có residual* thì chỉ 6.86 (không khác biệt có ý nghĩa). STID (7.43 ± 0.21) và DCRNN-lite (7.56–7.86) kém hơn ensemble GRU-RT có ý nghĩa.
4. **Đưa hồ sơ theo giờ vào mạng nơ-ron giúp ở tầm xa** (GRU-RTP 60 phút: 7.95 so với 8.39 của GRU-RT; T-GCN-RTSP 6.69) nhưng làm tệ đi ở 5 phút và **chưa có ý nghĩa thống kê** so với GRU-RT (p = 0.56 và 0.08).
5. **Chạy trên GPU (mục 11.8):** Graph WaveNet + residual tái lập đúng kết quả CPU (6.565 ± 0.052 so với 6.589 ± 0.117); **cỡ đầy đủ không tốt hơn cỡ gọn** (6.685 so với 6.565); STAEformer kém và rất không ổn định (cỡ đầy đủ 7.220 ± 0.431; cỡ gọn 7.628 ± 0.533 — còn kém T-GCN gốc). Trên 5 ngày dữ liệu, thêm tham số chỉ làm quá khớp nhanh hơn.

### 11.3 Phản biện — vì sao các mô hình SOTA không thắng ở đây

Các kiến trúc này được thiết kế và báo cáo trên METR-LA/PEMS-BAY (4–6 tháng). Los-loop chỉ có **5 ngày train** ≈ 1 400 cửa sổ:

- **Số tham số so với dữ liệu:** STID 117 nghìn, DCRNN-lite 67 nghìn, Graph WaveNet 166 nghìn so với 13–14 nghìn của GRU/T-GCN. Embedding thời gian của STID có 288 khung giờ × 32 chiều nhưng mỗi khung chỉ được thấy ~5 lần.
- **Dấu hiệu quá khớp đo được:** epoch tốt nhất của STID là 4–11, DCRNN-lite 25–27, GRU-RTP 6–11, T-GCN-RTSP 6–11 (GRU-RT: 19–37; GRU: 87–101), và độ lệch giữa các seed của STID lớn gấp 30 lần GRU (0.21 so với 0.006).
- **GPU xác nhận chẩn đoán này:** dùng cỡ của bài gốc (Graph WaveNet 301 nghìn, STAEformer 1.26 triệu tham số) không cải thiện mà còn tệ hơn trên Los-loop (Graph WaveNet + residual 6.565 → 6.685; STAEformer chọn epoch 2–15 và lệch giữa các seed 0.43–0.65, gấp 70–110 lần GRU), trong khi trên SZ-taxi (22 ngày) cỡ gọn và cỡ đầy đủ ngang nhau (4.428 so với 4.436). Giới hạn nằm ở *lượng dữ liệu*, không ở *sức tính* hay *kích thước mô hình*.
- **Ngoại lệ có tính hướng dẫn — Graph WaveNet + residual:** cũng quá khớp sớm (epoch tốt nhất 10–13) nhưng vẫn tốt: tích chập thời gian giãn nở + khuếch tán đồ thị hai chiều + đồ thị thích nghi cho biểu diễn đủ mạnh, còn residual (học Δ so với giá trị cuối — phát hiện ở mục 5.4) giữ cho mô hình không bao giờ tệ hơn persistence ở tầm ngắn. Bỏ residual: 6.86. Mẹo "đặt bài toán đúng" quan trọng ngang kiến trúc.
- **Kết luận đúng là "chưa kiểm chứng được trên dữ liệu đủ lớn", không phải "SOTA kém"** (mục 11.7 cho thấy STID và Graph WaveNet dẫn đầu trên SZ-taxi nhiều dữ liệu hơn): với METR-LA, Graph WaveNet/STAEformer giảm MAE 60 phút từ ~3.60 (DCRNN) xuống ~3.3–3.5. `tools/prepare_large_dataset.py` + `LOCAL_RUN.md` cho phép chạy lại toàn bộ trên các bộ đó.
- Ngược lại, LightGBM có **dừng sớm riêng cho từng tầm**: số cây tốt nhất khác hẳn nhau giữa các tầm (≈ 90 ở 5 phút, 200–550 ở tầm 15–45 phút, ≈ 50 ở 50–60 phút, seed 42), tức mỗi tầm tự chọn độ phức tạp phù hợp — còn mạng nơ-ron dùng một epoch chung cho cả 12 đầu ra. Cùng với việc cây quyết định khó học thuộc nhiễu hơn, đó đúng là thứ dữ liệu nhỏ cần.

### 11.4 Minh chứng — LightGBM dựa vào thông tin gì? (ảnh hưởng của thuộc tính)

![Hình 22. Đặc trưng quan trọng nhất (trái) và tỉ trọng từng nhóm thông tin theo tầm dự báo (phải)](figures/lgbm_importance_losloop.png)

Tỉ trọng gain theo nhóm thông tin (trung bình 3 seed):

LGBM_GROUPS

- **Hồ sơ theo giờ của chính cảm biến là nguồn thông tin số một** (~50% gain ở 5 phút, ~65% ở 60 phút): độ lệch hiện tại so với hồ sơ (`dev_from_prof`, "đang chậm/nhanh bất thường bao nhiêu") và thay đổi hồ sơ dự kiến tới t+k (`prof_change_k`, "sắp vào/ra giờ cao điểm"). Đây chính là thông tin mô hình học sâu thiếu: sin/cos của giờ cho biết *bây giờ là mấy giờ*, nhưng không cho biết *cảm biến này thường kẹt lúc mấy giờ*.
- **Láng giềng (đồ thị) đóng góp ~11–21%, đạt đỉnh ở tầm 15–25 phút** — khớp với kết quả mục 5 rằng đồ thị có ích nhất ở tầm trung/xa và khi biến động mạnh; ở 5 phút lịch sử của chính nút áp đảo.
- **Trạng thái toàn mạng và bất thường láng giềng (v2) thêm rất ít** (RMSE validation 5.352 → 5.348): thông tin đã nằm trong hồ sơ.
- **Siêu tham số gần như không quan trọng** — mọi cấu hình nằm trong 0.03 RMSE validation:

LGBM_TUNING

### 11.5 Giờ cao điểm: điểm yếu đã được khắc phục

MODERN_SUBSETS

- Lần đầu tiên có mô hình **thắng persistence trong giờ cao điểm ở 15 phút** (LightGBM 4.15, trộn 4.04 so với 4.20) và thắng rất xa ở 60 phút (trộn 5.88, LightGBM 5.99 so với 7.44 của persistence — giảm 19–21%; GRU-RT 8.02).
- Khi kẹt xe (v < 40 mph), MAE 60 phút giảm từ 13.3 (GRU-RT) xuống 10.4 (LightGBM): mô hình biết rằng cảm biến *đang trong khung giờ thường kẹt* nên không vội dự báo "sắp thông".
- **So với T-GCN gốc (dòng baseline)**: trong cao điểm, T-GCN gốc là mô hình kém nhất ở 15 phút (5.21; persistence 4.20); LightGBM giảm MAE 60 phút trong cao điểm từ 8.59 xuống 5.99 (−30%) và khi kẹt xe từ 16.2 xuống 10.4 (−36%). Mọi ô của LightGBM và bản trộn đều được in đậm.

### 11.6 Trộn theo tầm — vì sao thành viên "kém hơn" vẫn có ích

![Hình 23. Trọng số trộn theo tầm dự báo (fit trên validation)](figures/blend_weights_losloop.png)

BLEND_WEIGHTS

Trọng số tối ưu trên validation chia gần đều cho **LightGBM (~0.40) và Graph WaveNet-residual (~0.50)**, GRU-RT nhận ~0.05, T-GCN-RTSP gần 0. Hai mô hình đơn tốt nhất ở đây thuộc **hai họ rất khác nhau** — cây quyết định trên đặc trưng thiết kế tay và mạng nơ-ron đồ thị học biểu diễn trực tiếp — nên sai số của chúng ít tương quan và trộn lại giảm thêm 4–6% RMSE so với từng mô hình.

Bản trộn trước khi có Graph WaveNet (LightGBM + GRU-RT + GRU-RTP + T-GCN-RTSP) cho 6.29 / 3.58, với LightGBM 0.55 → 0.75 (tăng theo tầm) và GRU-RT 0.35 → 0.25; khi đó **GRU-RTP và T-GCN-RTSP — dù tốt hơn GRU-RT khi đứng một mình — nhận trọng số 0**, vì chúng dùng cùng thông tin hồ sơ như LightGBM nên sai số trùng nhau. Ensemble cần thành viên *đa dạng*, không cần thành viên mạnh nhất; vì vậy ensemble chia đều LightGBM + GRU-RTP (6.45) kém hơn trộn có trọng số.

So với bài báo (cách tính gộp của tác giả): mô hình trộn đạt PAPER_STYLE_BLEND.

### 11.7 SZ-taxi

Cùng các hướng trên SZ-taxi (22 ngày train, mục tiêu ≠ 0; Graph WaveNet và STAEformer chỉ chạy trên GPU, DCRNN không chạy):

SZ_MODERN_TABLE

![Hình 24. SZ-taxi: xếp hạng mọi mô hình (mục tiêu ≠ 0)](figures/modern_ranking_sztaxi.png)

Trên SZ-taxi, bức tranh **đảo ngược** so với Los-loop — và chính sự đảo ngược đó là minh chứng:

- **Các mô hình tự học hồ sơ theo giờ dẫn đầu:** Graph WaveNet chạy GPU (4.428 ± 0.011 / 3.131), STAEformer GPU (4.443 ± 0.016) và STID (4.445 ± 0.009 / 3.140) ngang nhau trong phạm vi ~2 độ lệch chuẩn giữa các seed, đều hơn LightGBM v2 (4.494) và GRU (4.619). Với STID, khác biệt so với GRU-RT (trên SZ-taxi chỉ có 1 seed) có ý nghĩa (p < 10⁻⁴); các run GPU chưa có tệp dự báo nên chưa kiểm định cặp được. Trộn theo tầm (STID 0.75–0.80, LightGBM 0.10–0.20) đạt **4.402 / 3.106**, giảm 4.7% RMSE và 5.0% MAE so với GRU, **6.2% / 7.2% so với T-GCN gốc**.
- **Vì sao STID thắng ở đây mà thua ở Los-loop?** Embedding "danh tính cảm biến" và "khung giờ" của STID thực chất **tự học hồ sơ theo giờ của từng cảm biến** — đúng tri thức mà LightGBM nhận qua đặc trưng thiết kế tay. Với SZ-taxi mỗi khung giờ được thấy ~22 lần trong train, đủ để học; với Los-loop chỉ ~5 lần, nên STID quá khớp. Cùng một ý tưởng, hai kết luận ngược nhau — phụ thuộc vào lượng dữ liệu.
- LightGBM v2 (thêm bất thường láng giềng/toàn mạng) cải thiện rõ trên SZ-taxi (4.494 so với 4.523 của v1), khác với Los-loop: đặc trưng trạng thái toàn mạng hữu ích hơn trên chuỗi nhiễu, nơi trung bình hóa nhiều cảm biến lọc được nhiễu.
- Residual tiếp tục không giúp trên dữ liệu nhiễu (STID + residual 4.449 ≈ STID 4.445; Graph WaveNet + residual 4.464 so với 4.428 trên GPU), nhất quán với mục 10 — ngược với Los-loop, nơi residual giúp chính Graph WaveNet (6.693 → 6.565 trên GPU). Trên chuỗi mượt, neo dự báo vào giá trị cuối là neo vào thông tin; trên chuỗi nhiễu là neo vào nhiễu.

### 11.8 Các lần chạy trên GPU: kiểm tra số liệu và kết quả

**Đã chạy gì.** 7 cấu hình (Graph WaveNet và STAEformer; cỡ gọn và cỡ đầy đủ theo bài gốc; có/không residual) × 3 seed (42, 123, 2026) × 2 bộ dữ liệu = 42 run, cùng giao thức nghiêm ngặt (70/10/20, thống kê chỉ từ train, dừng sớm theo validation; `python run_suite.py --dataset … --groups modern_gpu`). Các bảng bạn đẩy lên (sai số từng run; tóm tắt cho SZ-taxi) được giữ nguyên trong `reports/external/` để không script nào ghi đè, và `analysis/compare_models.py` gộp chúng với các run CPU vào `reports/tables/all_runs_*.csv` (106 run Los-loop, 62 run SZ-taxi).

**Kiểm tra số liệu** (`python analysis/gpu_results.py`):

GPU_CHECKS

- Tệp tóm tắt của bạn cho SZ-taxi khớp tuyệt đối (sai khác < 10⁻¹²) với giá trị tính lại từ 21 run — cả trung bình lẫn độ lệch chuẩn (ddof = 1), cho RMSE/MAE trên mọi mục tiêu và trên mục tiêu ≠ 0. Với Los-loop, bảng tóm tắt được dựng từ 21 run theo đúng cách đó.
- Ở mọi run, sai số tăng theo tầm dự báo — không có dấu hiệu lệch cột hay ghi nhầm tầm.
- **Graph WaveNet + residual tái lập giữa hai máy:** CPU 6.589 ± 0.117 và GPU 6.565 ± 0.052 (chênh 0.02, nhỏ hơn độ lệch chuẩn). Với Graph WaveNet không residual, run CPU duy nhất (seed 42: 6.860) cao hơn cả 3 run GPU (6.62–6.74) dù cùng seed và cùng chọn epoch 10: cùng một seed không cho cùng quỹ đạo huấn luyện trên phần cứng khác (phép cộng dấu phẩy động trên GPU không tất định), nên chênh lệch giữa hai máy cũng là một phần của độ biến thiên — thêm một lý do để mọi kết luận ở đây dựa trên 3 seed.

**Chỉ số kiểu bài báo cho các run GPU.** Bảng kết quả GPU chỉ có sai số tại từng bước, không có tệp dự báo. Vì mọi mô hình dùng chung các cửa sổ test (cùng số mục tiêu ở mỗi bước), sai số gộp các bước 1…k suy ra được **chính xác**: RMSE₁..ₖ = √(TB_j RMSE_j²), MAE₁..ₖ = TB_j MAE_j, Accuracy = 1 − √(n·RMSE₁..ₖ²) / ‖Y₁..ₖ‖ và R² = 1 − n·RMSE₁..ₖ² / SST(Y₁..ₖ), với Y là các mục tiêu test chung (n phần tử). Công thức được kiểm chứng trên các run CPU có cả hai loại số (khớp tới mọi chữ số in ra) và trên chính 42 run GPU (giá trị suy ra khi k = toàn bộ tầm trùng với RMSE, MAE, R² tổng mà run ghi lại: sai khác ≤ 3·10⁻¹⁵ và ≤ 6·10⁻¹¹). Nhờ vậy các dòng "(GPU)" có mặt trong bảng tổng hợp ở đầu báo cáo (`reports/tables/modern_gpu_cumulative_*.csv`).

**Kết quả — Los-loop:**

GPU_TABLE_LL

**Kết quả — SZ-taxi:**

GPU_TABLE_SZ

![Hình 25. Bảy cấu hình chạy trên GPU (trung bình ± độ lệch chuẩn 3 seed) so với các mốc chạy trên CPU](figures/modern_gpu_configs.png)

Phân tích:

1. **Los-loop: cỡ đầy đủ không giúp, STAEformer quá khớp.** Graph WaveNet cỡ đầy đủ (301 nghìn tham số) kém cỡ gọn cả khi không residual (6.752 so với 6.693) lẫn có residual (6.685 so với 6.565). STAEformer là kiến trúc kém nhất và kém ổn định nhất: cỡ gọn 7.628 ± 0.533 — **là cấu hình GPU duy nhất có RMSE kém T-GCN gốc (+1.5%)** — cỡ đầy đủ 7.220 ± 0.431; epoch tốt nhất chỉ 2–15, và seed tốt nhất (6.79–7.03) cách seed tệ nhất (7.65–8.10) 0.9–1.3 mph. Với ~1 400 cửa sổ train, attention theo thời gian và theo nút có quá nhiều bậc tự do (mục 11.3).
2. **Residual giúp ở Los-loop, hại ở SZ-taxi** — cùng kiến trúc, cùng máy: Graph WaveNet 6.693 → 6.565 trên Los-loop nhưng 4.428 → 4.464 trên SZ-taxi, đúng như kết luận về tính chất dữ liệu ở mục 10.
3. **SZ-taxi: kích thước gần như không quan trọng** — cả 7 cấu hình nằm trong 0.04 RMSE (4.428–4.464, mục tiêu ≠ 0) và ngang STID (4.445): với thông tin đầu vào hiện có, bộ dữ liệu nhiễu này có "trần" khoảng 4.40–4.45. Graph WaveNet cỡ gọn là mô hình đơn tốt nhất về số (4.428, −5.7% so với T-GCN gốc), nhưng chỉ hơn STID 0.017 (< 2 độ lệch chuẩn giữa các seed) nên chưa thể kết luận khi chưa có kiểm định cặp.
4. **Hạn chế:** chỉ có bảng sai số, chưa có tệp dự báo, nên chưa làm được kiểm định Diebold–Mariano, phân tích giờ cao điểm, conformal hay trộn theo tầm với các mô hình GPU. Nếu chép các thư mục `results/<dataset>/modern_gpu/<cấu hình>/seed_*/` (cần `metrics.json`, `history.csv`, `config.json`, `test_predictions.npz` và `best.pt` hoặc `val_predictions.npz`) vào repo (`git add -f`, vì `results/` bị bỏ qua), `analysis/modern_models.py` tự đưa chúng vào bảng và kiểm định (dưới tên `gpu_<cấu hình>`), còn `analysis/blend.py --members lgbm_best gpu_gwnet_res ...` đưa chúng vào bản trộn.

### 11.9 Mô hình cuối cùng mới và cách đọc lên để dự báo

Theo validation, **LightGBM (v2, 127 lá) là mô hình đơn tốt nhất**, **Graph WaveNet + residual là mạng nơ-ron tốt nhất**, và **trộn theo tầm hai mô hình này (+ GRU-RT với trọng số nhỏ) là dự báo cuối cùng** cho Los-loop. Các tệp trong `models/`: `losloop_lgbm/` (LightGBM), `losloop_gwnet_res.pt` (Graph WaveNet, seed tốt nhất theo validation), `losloop_gru_res_time_seed*.pt` (GRU-RT); SZ-taxi: `sztaxi_lgbm/`, `sztaxi_stid.pt` (Graph WaveNet chạy GPU nhỉnh hơn STID nhưng chưa có checkpoint trong repo, nên bản trộn SZ-taxi vẫn là STID-res + LightGBM). Mô hình LightGBM là một thư mục:

| Tệp | Nội dung |
|---|---|
| `model/h01.txt … h12.txt` | 12 mô hình cây tăng cường (một cho mỗi tầm 5…60 phút), định dạng văn bản của LightGBM |
| `model/stats.npz` | trung bình/độ lệch chuẩn train của từng cảm biến, ma trận kề chuẩn hóa, **hồ sơ theo giờ** [288 × 207], danh sách cảm biến, bước thời gian, phiên bản đặc trưng |

Đọc lên và dự báo (cùng lệnh `predict.py`, truyền thư mục thay vì tệp `.pt`):

```bash
python predict.py --checkpoint models/losloop_lgbm --input data/losloop/speed.csv --end-time "2012-03-07 07:30" --compare --sensors 4
```

Quy trình bên trong: đọc 12 mốc gần nhất → chuẩn hóa bằng thống kê train → dựng ~50 đặc trưng (độ trễ, láng giềng qua Â, giờ/thứ, độ lệch so với hồ sơ) → mỗi mô hình h_k dự báo Δ_k → ŷ_{t+k} = x_t + Δ_k → đổi về mph, cắt về ≥ 0. Kết quả dựng lại từ CSV khớp đúng dự báo đã lưu trên test (sai khác ~1e-5 mph).

PREDICT_GBDT_EXAMPLE

So với ví dụ cùng thời điểm ở mục 6.2: với cảm biến đang kẹt 767542 (thực tế 18–23 mph), ensemble GRU-RT dự báo "hồi phục" dần lên 38.5 mph sau 1 giờ, còn **LightGBM giữ ở 22–27 mph** — mô hình biết cảm biến này đang ở trong khung giờ thường kẹt (đặc trưng `dev_from_prof`, `prof_change_k`), nên không kéo dự báo về mức thông thoáng. Đây là minh chứng cụ thể cho việc giảm MAE khi kẹt xe ở mục 11.5.

Graph WaveNet được nạp đúng như GRU/T-GCN (`predict.py --checkpoint models/losloop_gwnet_res.pt ...`, checkpoint chứa kiến trúc, trọng số, ma trận chuyển P_f, P_b và thống kê chuẩn hóa). Trộn theo tầm không cần mã riêng: lấy dự báo của LightGBM, Graph WaveNet và GRU-RT rồi cộng có trọng số theo bảng ở mục 11.6 (trọng số lưu trong `results/losloop/ensemble/blend/metrics.json`).
