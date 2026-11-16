# Chốt cấu hình trước khi đo test

Ngày 16/11/2026. Corpus: 340 chunk Day 6, vector tạo lại bằng đúng embedding model `openai/text-embedding-3-small`, backend local cosine. Cổng pgvector Day 6 không mở.

| Dev, 15 câu có đáp án | TF-IDF k=3, b=3000 | Dense k=3, b=3000 | Hybrid k=3, b=3000 |
|---|---:|---:|---:|
| Hit@1 | 8/15 | 9/15 | **12/15** |
| Evidence Hit@3 | 13/15 | 13/15 | **14/15** |
| Context Evidence Rate | 13/15 | 13/15 | **14/15** |
| Agent có trả lời / 15 | chưa chạy | 14/15 | **15/15** |
| Agent từ chối đúng câu thiếu evidence / 2 | chưa chạy | 2/2 | 2/2 |

Chọn **hybrid RRF, top-3, budget 3000 ký tự, tối đa 2 lượt tìm, `openai/gpt-4o-mini-2024-07-18`**. So với dense, hybrid đưa evidence vào context thêm một câu và trả lời thêm một câu. Tuy nhiên audit thủ công trên dev cho thấy hybrid trả lời sai q07 (lấy số liệu Mỹ thay cho toàn cầu), và q05 gán thêm nguyên nhân không đúng phạm vi hỏi. Vì vậy test cần chấm nội dung và citation, không lấy trạng thái `answered` làm điểm đúng. Baseline test là dense với cùng k/budget/model. Không thay cấu hình sau khi xem test.

## Xác nhận trên pgvector Day 6 trước khi chạy test pgvector

Sau khi Docker Desktop chạy, volume database mới trống được nạp strategy C với đúng 340 chunk và vector/corpus/model đã kiểm chứng ở trên. Dev `--backend pgvector`: TF-IDF Hit@1 8/15, Evidence Hit@3 13/15; dense 9/15 và 13/15; hybrid 12/15 và 14/15. Agent pgvector dense/hybrid đều dùng tool 17/17, trả lời 15/15 câu có đáp án và từ chối đúng 2/2 câu thiếu evidence theo trạng thái. Audit phát hiện dense q02 trả lời **40% thay vì 31%**, còn hybrid q02 đúng; hybrid vẫn sai q07 và thêm nguyên nhân sai phạm vi ở q05. Giữ **hybrid** làm cấu hình đã chốt cho test pgvector; dense là baseline. Đây là lượt thí nghiệm pgvector riêng, không dùng kết quả test local để đổi tham số.
