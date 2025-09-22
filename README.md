# Travel AI – Bài toán gợi ý điểm đến

## Input

- `city` (tùy chọn): thành phố muốn đi.
- `days` (bắt buộc): số ngày dự kiến.
- `budget` (bắt buộc): ngân sách dự kiến (VND).
- `prefs_text` (bắt buộc): mô tả sở thích dạng text (ví dụ: "biển, hiking, ẩm thực").

## Output

- Top-k điểm đến: mỗi item gồm `{id, name, score, reason}`.

## Ràng buộc

- **Cứng**: nếu có `city` thì chỉ xét điểm đến thuộc `city` đó.
- **Mềm**: cho phép lệch `budget` và `days` một chút (penalty càng lớn khi lệch nhiều).

## Xếp hạng

- Điểm cuối: `final = 0.6 * cosine + 0.25 * budget_score + 0.15 * duration_score`
  - `cosine`: độ phù hợp văn bản giữa `prefs_text` và mô tả điểm đến.
  - `budget_score`: 1.0 nếu chi phí điểm đến gần `budget`, giảm dần khi lệch xa.
  - `duration_score`: 1.0 nếu số ngày phù hợp, giảm dần khi lệch.
- Chuẩn hóa về [0,1] trước khi cộng trọng số.

## API

- `POST /rank` với JSON:

```json
{
  "city": "Da Nang",
  "days": 3,
  "budget": 3000000,
  "prefs_text": "biển, check-in, ẩm thực hải sản",
  "top_k": 5
}
```
