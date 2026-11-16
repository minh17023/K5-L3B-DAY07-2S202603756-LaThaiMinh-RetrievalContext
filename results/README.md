# Kết quả đã đo

Các file `retrieval-...-{summary,traces}.json` ghi ba phương pháp trên dev và dense/hybrid trên test. `agent-...-{summary,traces}.json` ghi agent thật dùng OpenRouter, query tool, context, câu trả lời/citation, token và thời gian. Bộ test gồm 5 câu có đáp án và 2 câu thiếu evidence. Các file `reviews-test-*.json` là nhãn audit thủ công từng case; `agent-test-*-metrics.json` do `starter/agent_metrics.py` tổng hợp.

**Thứ tự quan trọng:** [dev-selection.md](./dev-selection.md) ghi việc chọn hybrid trước test local và trước test pgvector. Đọc [báo cáo](../report/individual/2A202603756_LaThaiMinh.md) để xem lệnh chạy và cách chấm. Các file `local` dùng cosine trên cache vector; các file `pgvector` truy vấn trực tiếp PostgreSQL Day 6 strategy C đã nạp lại 340 chunk. Volume Docker cũ không hiện diện; volume mới được tạo và nạp từ corpus/vector Day 6 đã xác minh.
