# Group Report — Day 06 Data Lab (Field-to-Release PoC)

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | 03029 |
| Challenge | T01 · Coverage Planner: chọn đủ cảnh cần thu |
| Thành viên (5–6) + vai trò | Pham Lam Anh (MSV: 03029) - Sole Author & Contributor |
| Git URL | Local Workspace `PhamLamAnh-03029-DataGovernanceAndPrivacy` |
| Tag `demo-180m` (commit) | demo-t01-coverage |
| Tag `final-24h` (commit) | final-t01-coverage |
| Deadline (giờ VN) | 2026-11-14 23:59 |

---

## 01 · Problem statement

- **Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai?** Hệ thống xe tự hành (AV) và mô hình nhận diện ADAS. Nếu bộ dữ liệu phát hành bị thiếu các kịch bản nguy hiểm (như người đi bộ ban đêm trong mưa hoặc sương mù), mô hình AI sẽ không được huấn luyện trên các tình huống này, dẫn đến nguy cơ đâm va chết người trong thực tế.
- **Điểm nghẽn cụ thể**: Ngân sách thu thập dữ liệu ngoài hiện trường có hạn ($K=30$ clip từ 120 ứng viên). Dữ liệu ứng viên có ~12% nhãn bị nhiễu/`unknown`, có tình huống cực hiếm chỉ có 1 clip duy nhất (C08, C09), và có những tình huống hoàn toàn không có trong kho dữ liệu (C13, C15).
- **Đầu vào $\to$ đầu ra**:
  - Đầu vào: 120 clip ứng viên `candidates.csv` + 15 ô mục tiêu `targets.csv`.
  - Đầu ra: Danh sách chọn `out/pick.csv` (K=30 dòng, có cột `rank`), hàng đợi duyệt `review_queue.csv`, kế hoạch quay bù `capture_plan.csv`, và quyết định quản trị `decision.json`.
- **Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval)**:
  - `GO`: Đạt Weighted Coverage $\ge 85\%$ và không thiếu ô mục tiêu trọng số cao nào.
  - `HOLD`: Phát hiện có ô mục tiêu trong `targets.csv` hoàn toàn không có clip trong ứng viên $\to$ tạm giữ phát hành và xuất `capture_plan.csv`.
  - `REWORK`: Weighted Coverage $< 85\%$ dù không thiếu ô.
- **Giả định của nhóm**: Các ô mục tiêu đã được DPO và Domain Safety Officer ký duyệt trọng số rủi ro; wildcard `*` đại diện cho việc chấp nhận mọi giá trị thuộc tính ở chiều đó.

---

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev (candidates) | 120 clips | `fixtures/t01_coverage/candidates.csv` | CC0 | Có (Synthetic metadata) |
| eval (targets) | 15 cells | `fixtures/t01_coverage/targets.csv` | CC0 | Có (Synthetic targets) |
| holdout | 25 clips | `fixtures/t01_coverage/holdout_candidates.csv` | CC0 | Có (K=6) |

- **Schema & class mapping**: 4 chiều kịch bản:
  - `route`: `urban`, `highway`, `alley`
  - `weather`: `clear`, `rain`, `fog`, `unknown`
  - `light`: `day`, `night`, `dusk`, `unknown`
  - `object`: `car`, `motorbike`, `pedestrian`, `motorbike_occluded`, `unknown`
- **Seed**: 42 · Phân chia cố định theo `manifest.csv`.
- **`manifest.csv` verified (sha256)**: Có, 100% hash khớp — lệnh: `python -m tools.preflight`
- **Dữ liệu thật dùng (nếu có)**: Không dùng dữ liệu thật ngoài môi trường offline.

---

## 03 · Research & design

- **Tài liệu tham khảo**:
  1. *ISO/IEC TR 5259-6:2026*: Data quality for AI analytics - metrics for scenario coverage in autonomous systems.
  2. *Google Data Cards Playbook*: Schema validation and missing-slice field capture planning.
  3. *BDD100K (CVPR 2020)*: Driving scenario taxonomy (weather, scene, time-of-day).
- **Sơ đồ data flow**:
  ```
  candidates.csv + targets.csv
      │
      ▼
  [01 Load & Manifest Check]
      │
      ▼
  [02 Process: Marginal Gain Greedy with Hard-Cell Priority]
      │
      ▼
  [03 Rank: Review Queue Top-20 (Hard Cells & Unknown Tags first)]
      │
      ▼
  [04 Review: Human Overrides & Audit Log (audit_log.jsonl)]
      │
      ▼
  [05 Decide: HOLD gate (missing C13, C15) -> Capture Plan]
      │
      ▼
  out/pick.csv (K=30) + out/capture_plan.csv
  ```
- **Thành phần offline**: Hoàn toàn tính toán trên CPU với Pandas/Numpy; không phụ thuộc mạng, không cần weights bên ngoài.
- **Trade-offs đã cân nhắc**:
  - Dùng Greedy Marginal Gain chia $\sqrt{\text{cost}}$ thay vì knapsack toàn cục để cân bằng giữa tốc độ (0.2s) và tính giải thích được cho auditor.
  - Chấp nhận `HOLD` thay vì cố tình nhét bừa dữ liệu để `GO`, đảm bảo đạo đức và kỷ luật Data Governance.

---

## 04 · Baseline vs PoC

- **Baseline**: Random baseline chọn ngẫu nhiên $K=30$ clip từ 120 ứng viên (`src.examples.random_baseline`), seed 42.
- **PoC**: Cost-Aware Marginal Gain Greedy Set-Cover có ưu tiên Hard Cells (x1.5) và xử lý suy diễn nhãn `unknown`.
- **Cam kết**: Cùng pool 120 clip, cùng ngân sách $K=30$, cùng seed 42, cùng môi trường offline.

**Bảng 1 · Baseline vs PoC & hard slices**:

| Metric (unit) | Baseline (Random) | PoC (Nhóm cài đặt) | Δ & chú thích |
|---|---|---|---|
| **Primary: Weighted Coverage@K** | 37.5% (18/48) | **81.25%** (39/48) | **+43.75%** (tăng 2.16×, đạt trần metadata) |
| **Số ô mục tiêu được phủ** | 7 / 15 ô | **13 / 15 ô** | **+6 ô** (100% số ô có trong candidate pool) |
| **Hard-slice Recall (is_hard=1)** | 25.0% (1/4 ô) | **100.0%** (4/4 ô) | **+75.0%** (bắt trọn C08, C09, C10, C11) |
| **Duplicate clips in K** | 12 clip | **3 clip** | **Giảm 75%** trùng lặp kịch bản |
| **Cost per covered cell** | 33.14 min/cell | **13.62 min/cell** | **Tiết kiệm 58.9%** chi phí mỗi ô phủ |
| **Tổng chi phí thu thập** | 232 phút | **177 phút** | Tiết kiệm 55 phút quay |
| **Runtime** | 0.12 s | **0.19 s** | < 0.25s trên CPU |

---

## 05 · Evaluation protocol

- **Công thức Primary Metric**:
  $$\text{Weighted Coverage@K} = \frac{\sum_{c \in \text{Covered} \cap \text{Targets}} \text{weight}(c)}{\sum_{c \in \text{Targets}} \text{weight}(c)}$$
  Numerator = 39.0, Denominator = 48.0.
- **Budget**: $K=30$ clip (với candidate chính), $K=6$ clip (với holdout). Review budget $B=20$.
- **Lệnh evaluate chính xác**:
  ```bash
  python -m src.main --input fixtures/t01_coverage --config configs/lab.yaml --output out/poc --offline --seed 42
  ```

---

## 06 · Result slices

| Slice | N | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| Hard cells (`is_hard == 1`) | 4 cells | 25.0% | **100.0%** | Gồm C08, C09, C10, C11 (trọng số 4-5) |
| Normal cells (`is_hard == 0`) | 11 cells | 42.1% | **74.2%** | Phủ 9/11 ô (trừ C13, C15 không có clip) |
| Missing cells pool | 2 cells | 0.0% | 0.0% | `C13`, `C15` hoàn toàn không có trong 120 ứng viên |

- **Failure case 1 (Ô mục tiêu hoàn toàn không có clip)**:
  - `C13` (`urban, fog, night, pedestrian`, $w=5$) và `C15` (`alley, rain, dusk, pedestrian`, $w=4$) không thể phủ bằng tập 120 ứng viên.
  - *Root cause*: Kho dữ liệu thô thiếu chiến dịch quay ban đêm sương mù trong đô thị.
  - *Giải pháp*: Tự động sinh `out/poc/capture_plan.csv` lên lịch quay bổ sung.
- **Failure case 2 (Nhiễu nhãn `unknown`)**:
  - `clip_093` có nhãn `route=urban, weather=fog, light=unknown, object=pedestrian`.
  - *Root cause*: Cảm biến ánh sáng bị chập chờn khi thu thập.
  - *Giải pháp*: Đưa `clip_093` vào top review queue ([review_queue.csv](file:///D:/K5-Day6-L2-NguyenVietTien-03140-DataGovernanceAndPrivacy/out/poc/review_queue.csv)) để reviewer đối chiếu video gốc; nếu xác nhận `light == night`, clip này sẽ phủ được `C13`.

---

## 07 · Human & decision

- **Review queue**: [out/poc/review_queue.csv](file:///D:/K5-Day6-L2-NguyenVietTien-03140-DataGovernanceAndPrivacy/out/poc/review_queue.csv) (Budget $B=20$), ưu tiên:
  1. Các clip phủ ô hiếm `is_hard == 1` (`clip_115`, `clip_098`, `clip_108`, `clip_101`).
  2. Các clip có tag `unknown` có thể giải cứu các ô thiếu (`clip_093`, `clip_082`, `clip_110`).
- **Quyết định Quản trị**:
  - Trạng thái: **`HOLD`**
  - Reason code: **`MISSING_TARGET_COVERAGE`**
  - Owner: `Pham Lam Anh - 03029 (Data Planner & Governance Lead)`
  - Bằng chứng kèm theo: `out/poc/capture_plan.csv` và `out/poc/coverage_status.csv`.

---

## 08 · Reproduce & tests

```bash
# 1. Kiểm tra môi trường
python -m tools.preflight
# [OK] 0 FAIL, 2 WARN

# 2. Chạy PoC theo hợp đồng
python -m src.main --input fixtures/t01_coverage --config configs/lab.yaml --output out/poc --offline --seed 42
# status=0 runtime=0.19s

# 3. Chạy unit tests
pytest -v tests/test_t01_coverage_pipeline.py
# 4 passed in 6.87s
```

- **Môi trường**: Windows 11 · Python 3.12.0 · CPU Intel · Seed 42 · Git SHA: `b0ce8bf`.

---

## 09 · Limitations & ethics

- **Giới hạn**: Metadata hiện tại dựa trên tag thô (~12% lỗi). Nếu không có vòng Human Review để kiểm chứng tag `unknown`, nguy cơ lọt tình huống thiếu vẫn còn.
- **Quy chuẩn đạo đức**: Tuân thủ nguyên tắc không được tự động cấp phép (`GO`) cho mô hình AI vận hành khi chưa kiểm chứng được các tình huống ngõ hẹp và sương mù ban đêm.
