# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T01 · Coverage Planner |
| Primary metric (tên đúng như evaluator in ra) | Weighted Coverage@K |
| Tăng tốt hay giảm tốt? | higher (tăng = tốt) |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | target cell |
| Công thức + numerator / denominator | Σ weight(covered target cells) / Σ weight(all target cells) |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | exact match trên 4 chiều `[route, weather, light, object]` (với `*` là wildcard) |
| Candidate pool cố định | `fixtures/t01_coverage/candidates.csv` (120 clips) · holdout: 25 clips |
| Review budget B / K | K = 30 (holdout K = 6) · Review budget B = 20 |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | Greedy Marginal Gain với trọng số rủi ro (hard cells x1.5) chia cho sqrt(cost) |
| Eval split (không dùng để tune) | `fixtures/t01_coverage/targets.csv` (15 cells) |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | Rare-scenario Recall@K (bỏ sót ô hard), Duplicate clips in K |
| Slices sẽ báo cáo (≥2) | 1. Hard cells slice (`is_hard == 1`: C08, C09, C10, C11) · 2. Missing cells slice (C13, C15) |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Exact cell count reporting (N=15 target cells) |
| Ngày giờ chốt + người chốt | 2026-11-14 · Pham Lam Anh - 03029 (Data Planner & Governance Lead) |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.
