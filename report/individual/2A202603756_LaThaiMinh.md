# Báo cáo Day 7

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** World Bank, *Commodity Markets Outlook, October 2026*, [PDF chính thức](https://thedocs.worldbank.org/en/doc/2ac43f46629cf90c2ea5b8929a343593-0050012026/original/CMO-October-2026.pdf), giấy phép CC BY 3.0 IGO; thành phần bên thứ ba có thể có quyền riêng. PDF 70 trang vật lý, **58 trang nội dung** (trang 9–66). SHA-256 `981338d84c7fd0d6d2c8ce504d5339c5ef721b4961c5a5b90945f40666714c22` khớp [manifest Day 6](../../data/day6_manifest.json). [prepare.py](../../src/prepare.py) dùng logic [day6_pipeline/](../../src/day6_pipeline/) để tái tạo **340 chunk B/C**, tối đa 900 ký tự, không vượt trang, giữ chunk ID, source ID, trang PDF, section và metadata. Corpus trong [chunks.json](../../data/chunks.json). PDF được Git bỏ qua.
- **Vector store, embedding model, phiên bản và số chiều:** Docker Compose Day 6 chạy `pgvector/pgvector:0.8.7-pg16` (PostgreSQL 16) tại `127.0.0.1:55433`; container `starter-postgres-1` healthy. Do volume cũ không hiện diện, volume `starter_pg_data` mới được nạp strategy C từ corpus/vector Day 6 đã xác minh bằng [seed_pgvector.py](../../src/seed_pgvector.py); SQL xác nhận `C|openai/text-embedding-3-small|340`. Đây là **index tái tạo**, không phải volume cũ được khôi phục. Embedding OpenRouter `openai/text-embedding-3-small`, **1536 chiều**, 56.689 document embedding tokens. [retrieval.py](../../src/retrieval.py) thực sự gọi `PgVectorStore.search(strategy="C")`, exact cosine qua pgvector; cũng có fallback cosine local trên cùng vector.
- **Agent model, cách chạy và phân chia dev/test:** [agent.py](../../src/agent.py) gọi OpenRouter Chat Completions với `openai/gpt-4o-mini-2024-07-18` và tool `search_report`. Benchmark [cases.json](../../data/cases.json) có **24 câu**: dev 15 có đáp án (`q01`–`q15`) + 2 thiếu evidence; test 5 có đáp án mới (`t01`–`t05`) + 2 thiếu evidence. Dev kế thừa câu có đáp án Day 6; test mới lấy evidence ở trang PDF 10, 29, 35, 50, 58. Không dùng `q16`–`q20` Day 6 làm test vì kết quả cũ đã lộ. [questions_answerable.json](../../data/questions_answerable.json) chứa 20 câu có gold cho retrieval. Gold/đáp án chuẩn không đi vào prompt hay tool.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** 20 gold có đúng `source_id`, trang và chuỗi evidence nằm trọn trong chunk; [build_benchmark.py](../../src/build_benchmark.py) kiểm tra điều này. Bốn câu thiếu evidence hỏi dự báo ngoài niên hạn hoặc mức chi tiết báo cáo không cung cấp, đã rà toàn corpus. Retrieval tính Hit@1, Evidence Hit@3 và Context Evidence Rate chỉ trên câu có đáp án (N=15 dev; N=5 test). Evidence hit cần đúng nguồn, trang và nội dung gold. Agent được audit thủ công: 0 = sai/từ chối nhầm; 1 = đúng một phần hoặc thêm khẳng định sai; 2 = đúng đầy đủ. Mỗi citation được đối chiếu với passage, câu thiếu evidence được chấm abstention; lỗi provider tách riêng.

## Retrieval và thử nghiệm

Ba phương pháp dùng cùng 340 chunk: **dense** query embedding + pgvector strategy C; **TF-IDF** từ [baseline.py](../../starter/baseline.py); **hybrid** lấy tối đa 12 ứng viên mỗi nhánh rồi hợp nhất bằng RRF `1/(60+rank)`, khử trùng ID. Filter `section` chỉ áp dụng khi case cung cấp. Không dùng rerank trong các cấu hình dưới. Budget là **ký tự context cuối**, không phải token. `n/N` ở các cột chất lượng là số câu có evidence đúng trên toàn bộ câu có đáp án của split.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Baseline | Dense pgvector C, k=3, filter theo case, không rerank, 3000 ký tự | Mốc vector Day 6 | dev | 9/15 (60,0%) | 13/15 (86,7%) | 13/15 (86,7%) | 1.065,44 ms/query; 266 query embedding tokens |
| Phương pháp 2 | TF-IDF, k=3, filter theo case, không rerank, 3000 ký tự | Lexical không gọi embedding API | dev | 8/15 (53,3%) | 13/15 (86,7%) | 13/15 (86,7%) | 2,73 ms/query; 0 token API |
| Phương pháp 3 | Hybrid RRF, k=3, filter theo case, không rerank, 3000 ký tự | Kết hợp semantic/từ khóa | dev | **12/15 (80,0%)** | **14/15 (93,3%)** | **14/15 (93,3%)** | 1.220,80 ms/query; 266 query embedding tokens |
| Mốc context thấp | TF-IDF local, k=3, 800 ký tự | Kiểm tra mất evidence khi đóng gói | dev | 8/15 (53,3%) | 13/15 (86,7%) | 3/15 (20,0%) | 3,75 ms/query; 0 token API |
| Cải thiện 1 | TF-IDF local, k=3, 3000 ký tự | Tăng budget 800 → 3000 | dev | 8/15 (53,3%) | 13/15 (86,7%) | **13/15 (86,7%)** | 3,07 ms/query; 0 token API |
| Cải thiện 2 | TF-IDF local, k=5, 3000 ký tự | Thêm ứng viên, kiểm q08 hạng 4 | dev | 8/15 (53,3%) | 13/15 (86,7%) | 13/15 (86,7%) | 4,04 ms/query; 0 token API |
| **Cấu hình đã chọn** | **Hybrid pgvector C + TF-IDF, k=3, filter theo case, không rerank, 3000 ký tự** | Chốt theo dev trước test | test | 3/5 (60,0%) | **4/5 (80,0%)** | **4/5 (80,0%)** | 1.059,74 ms/query; 109 query embedding tokens |
| Baseline | Dense pgvector C, k=3, filter theo case, không rerank, 3000 ký tự | Cùng test và tham số | test | 3/5 (60,0%) | 3/5 (60,0%) | 3/5 (60,0%) | 1.031,93 ms/query; 109 query embedding tokens |

**Hai thay đổi đã thử:** tăng budget từ 800 lên 3000 ở TF-IDF k=3 nâng Context Evidence Rate **3/15 → 13/15**, nhưng không thay Hit@1/3. Nhiều chunk gần 900 ký tự cộng header không vừa budget 800. Tăng k=3 lên k=5 không nâng ba metric: q08 có gold ở hạng 4, nên vẫn không được tính vào Evidence Hit@3; thời gian lần đo 3,07 → 4,04 ms nhưng dao động nhỏ không đủ khẳng định khác biệt tốc độ. Dense/hybrid bao gồm độ trễ API embedding query, còn TF-IDF không có API, nên không so thời gian như cùng hạ tầng.

**Hai lỗi dev và nguyên nhân:** q07 hỏi biến động sản lượng khí **toàn cầu** năm 2026. TF-IDF không có gold trang 29 trong top-5; hybrid trên câu gốc cũng không có trong top-3. Agent hybrid truy được đoạn trang 28 nói về **Mỹ** rồi dùng mức tăng 4%/22 bcm của Mỹ thay cho mức **giảm 0,8%/35 bcm toàn cầu** ở trang 29. Citation ID tồn tại nhưng câu trả lời sai phạm vi. q08 hỏi nhu cầu than 2026: TF-IDF xếp gold trang 31 hạng 4; tăng k=5 chỉ đưa evidence vào danh sách ứng viên, không cải thiện Hit@3. Hybrid trả lời đúng từ trang 31. Audit thêm q05 cho thấy model gán các yếu tố làm **Brent** tăng thành nguyên nhân khiến **Gulf voyage rates** tăng; nguyên nhân trực tiếp trong gold chỉ là thiếu tàu chở dầu.

**Lý do chọn:** hybrid dev pgvector Hit@1 **12/15** so với dense **9/15**, Evidence Hit@3 **14/15** so với **13/15**, Context Evidence Rate cùng mức cải thiện. Trong trace dev pgvector, dense q02 trả lời **40%** thay vì gold **31%**, hybrid trả lời đúng; hybrid vẫn có lỗi q07/q05 nêu trên. Cấu hình hybrid k=3, budget 3000, tối đa 2 lượt search được ghi trong [dev-selection.md](../../results/dev-selection.md) **trước khi chạy test pgvector**. Các phép đo local trước đó cho cùng Hit@1/Hit@3 và được lưu riêng trong [results/](../../results/); không lấy kết quả local thay cho pgvector. Chi phí USD không ghi số giả vì chưa xác minh đơn giá provider/model tại thời điểm gọi; token thật có trong summary/trace.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** Model phải gọi `search_report` trước khi trả lời, tự viết query và có thể tìm lại, tối đa **2 lượt search**. Tool chạy dense hoặc hybrid đã chốt, lấy top-3 và đóng gói context tối đa **3000 ký tự**; mỗi passage có chunk ID, nguồn và trang. Citation phải trùng ID trong context, nếu thiếu/sai thì agent từ chối. Khi không đủ evidence, trả “Insufficient evidence in the report.” Lỗi API/vector store được ghi `status=error` riêng, không tính như abstention. Trace giữ query, kết quả/rank/score, context, câu trả lời, citation, `rejected_answer`, token và thời gian. Sau dev smoke, prompt bỏ citation mẫu `[document:page:chunk]` vì model đã sao chép mẫu; smoke sửa xong trả ID thật.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---:|---:|---|---:|---:|---:|---:|
| Dense pgvector / dev | 17/17 | 19 search | Chỉ audit lỗi q02, chưa gán nhãn đủ 15 câu | Chưa audit đủ | 15/15 câu trả lời | 2/2 | 0/15 theo status; q02 trả lời sai |
| Hybrid pgvector / dev | 17/17 | 19 search | Chỉ audit lỗi q05/q07, chưa gán nhãn đủ 15 câu | Chưa audit đủ | 15/15 câu trả lời | 2/2 | 0/15 theo status; q07 trả lời sai |
| **Hybrid pgvector / test** | **7/7** | **11 search** | **0: 1; 1: 0; 2: 4; TB 1,6/2** | **5/5** | **4/4** | **2/2** | **1/5** |
| Dense pgvector / test | 7/7 | 10 search | 0: 1; 1: 1; 2: 3; TB 1,4/2 | 3/5 | 4/4 | 2/2 | 1/5 |

Hai lượt test đều có **0 lỗi provider**. `Tool use` tính số case thực sự gọi search; “Số lượt gọi” là tổng các search event, không phải số HTTP chat completion. Ở dev, trạng thái `answered` không được coi là grounded answer 2; bảng ghi rõ phần chưa gán nhãn đầy đủ để tránh suy diễn.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q05 | Dev có evidence | Hybrid tìm trang 22, 23, 22 | Nêu tanker shortages **và** các yếu tố khác; citation trang 22 | Đúng một phần: các yếu tố khác thuộc giá Brent, không trực tiếp là nguyên nhân voyage rates. |
| q07 | Dev có evidence | Hybrid tìm trang 14, 28, 28; thiếu gold trang 29 | Dùng số **Mỹ +4%/22 bcm** với citation trang 28 | 0: sai phạm vi so với **toàn cầu −0,8%/35 bcm**. |
| t01 | Test có evidence | Hai query đều tự thêm “October 2023”; các trang trả 22/10/24 rồi 22/10/23 nhưng context không có con số | “Insufficient evidence in the report.”; không citation | 0 cho cả hai: từ chối nhầm, gold **100 triệu thùng**. |
| t02 | Test có evidence | Hybrid tìm lại, lần hai có trang 29 | **+2,5% (110 bcm)**; citation chunk trang 29 | 2: đúng số, đơn vị và nguồn. |
| t03 | Test có evidence | Hybrid lấy trang 35, 35, 14 | **Lớn thứ hai; cao hơn khoảng 3%**; citation trang 35 | 2: đúng và được nguồn hỗ trợ. |
| t04 | Test có evidence | Hybrid lấy trang 50, 50, 52 | **Đồng +33% y/y**; citation trang 50 | 2: đúng; phần giảm 2027/2028 cũng có trong passage. |
| t05 | Test có evidence | Hybrid lấy trang 60, 62, 59; strict gold ở trang 58 không hit | Kim loại công nghiệp/thực phẩm truyền, gia súc/soft commodities nhận; citation trang 62 và 59 | Hybrid 2, citation **2/2** hỗ trợ. Dense 1: lẫn vai trò bạc riêng lẻ với nhóm precious metals; hai citation không hỗ trợ toàn bộ claim. |
| t_missing_01 | Test thiếu evidence | Hai lượt tìm giá urea tháng 12/2031 | Từ chối, không citation | Đúng: báo cáo không dự báo mức giá chi tiết đó. |
| t_missing_02 | Test thiếu evidence | Hai lượt tìm giá lithium 2032 | Từ chối, không citation | Đúng: báo cáo không dự báo năm đó. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** [agent-test-pgvector-hybrid-k3-b3000-s2-traces.json](../../results/agent-test-pgvector-hybrid-k3-b3000-s2-traces.json) và [agent-test-pgvector-dense-k3-b3000-s2-traces.json](../../results/agent-test-pgvector-dense-k3-b3000-s2-traces.json) lưu từng query/tool/context/answer. Audit thủ công trong [reviews-test-pgvector-hybrid.json](../../results/reviews-test-pgvector-hybrid.json) và [reviews-test-pgvector-dense.json](../../results/reviews-test-pgvector-dense.json); [agent_metrics.py](../../starter/agent_metrics.py) tạo [hybrid metrics](../../results/agent-test-pgvector-hybrid-metrics.json) và [dense metrics](../../results/agent-test-pgvector-dense-metrics.json). Cách chạy được ghi ngay dưới đây. Test agent dense: **10.100 prompt / 615 completion tokens**, trung bình **5.488,5 ms/case**; hybrid: **10.618 / 609 tokens**, **5.808,4 ms/case**. Dev dense: 19.051 / 1.549 tokens, 5.078,3 ms/case; dev hybrid: 18.823 / 1.525 tokens, 5.333,8 ms/case. Không suy đoán phí USD khi chưa có đơn giá đã xác minh.
- **Kết quả trước/sau, lỗi, đánh đổi và giới hạn:** Hybrid pgvector test cải thiện Evidence Hit@3 **3/5 → 4/5** và câu đạt grounded answer 2 **3/5 → 4/5** so với dense, đổi lại retrieval trung bình tăng **1.031,93 → 1.059,74 ms/query** trong hai lượt đo; chênh lệch nhỏ không đủ kết luận thống kê. Cả hai từ chối nhầm t01 do query drift; hybrid dev q07 sai phạm vi. Chỉ kiểm tra citation ID hợp lệ không phát hiện các lỗi này. Docker volume Day 6 được **tái tạo** từ dữ liệu đã kiểm chứng, không khôi phục volume cũ. Hash PDF, 340 chunk, 20 gold, 24 case, SQL count và **19/19 unittest** đã xác minh; `git diff --check` sạch. `.env`, PDF và vector cache không được commit.

## Thử thách nâng cao (nếu thực hiện)

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:** Không thực hiện bonus. Corpus hiện có **58 trang nội dung**, không đạt điều kiện tài liệu từ 100 trang.
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:** Không áp dụng cho bonus; phần cơ bản index text theo chunk Day 6.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:** Không tạo benchmark bonus; benchmark 24 case ở trên chỉ dành cho phần cơ bản.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---:|---:|---|
| Bảng | — | — | Không thực hiện bonus |
| Ảnh, sơ đồ | — | — | Không thực hiện bonus |
| Biểu đồ | — | — | Không thực hiện bonus |
| Multi-hop | — | — | Không thực hiện bonus |
| Text | — | — | Không thực hiện bonus |
| Thiếu bằng chứng | — | — | Không thực hiện bonus |
| **Toàn bộ** | **—** | **—** | **Không nhận điểm bonus** |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:** Không áp dụng cho bonus; trace phần cơ bản ở [results/](../../results/).
- **Giới hạn, cách tái lập và đường dẫn benchmark:** Điều kiện tài liệu từ 100 trang chưa đạt. Cách tái lập phần cơ bản ở mục dưới; benchmark ở [cases.json](../../data/cases.json).

## Cách chạy lại phần cơ bản

Chạy từ thư mục gốc repo bằng Python 3.11+; cài `pip install -r requirements.txt`. Tạo `.env` từ `.env.example`, điền `OPENROUTER_API_KEY` và `DATABASE_URL` trỏ tới pgvector Day 6 tại `127.0.0.1:55433`. Khởi động Docker Compose trong repo Day 6. Để tái tạo chunk, tải PDF theo `exact_download_url` trong [manifest](../../data/day6_manifest.json) vào `data/CMO-October-2026.pdf`; [prepare.py](../../src/prepare.py) kiểm tra hash rồi tạo 340 chunk. Repo đã có [chunks.json](../../data/chunks.json), nên có thể bỏ qua bước này nếu dùng dữ liệu nộp sẵn.

```powershell
python -m unittest discover -s tests -v
python -m src.prepare
python -m src.build_benchmark
python -m src.run index
python -m src.seed_pgvector
python -m src.run retrieval --backend pgvector --split dev --method dense
python -m src.run retrieval --backend pgvector --split dev --method tfidf
python -m src.run retrieval --backend pgvector --split dev --method hybrid
python -m src.run agent --backend pgvector --split dev --method hybrid --max-searches 2
python -m src.run retrieval --backend pgvector --split test --method hybrid
python -m src.run agent --backend pgvector --split test --method hybrid --max-searches 2
```

`index` tạo cache vector cục bộ bằng `openai/text-embedding-3-small` và tốn API tokens; `seed_pgvector` chỉ chạy trên strategy C trống, sau khi có PDF đúng hash. Nếu database Day 6 đã có đúng 340 chunk strategy C, bỏ qua cả hai lệnh này. Lệnh test chỉ chạy sau khi chốt cấu hình trên dev theo [dev-selection.md](../../results/dev-selection.md). Có thể thay `hybrid` bằng `dense` để chạy baseline test; mỗi lượt tạo summary và trace trong [results/](../../results/). Chấm câu trả lời và citation theo passage, ghi nhãn vào `reviews-test-pgvector-*.json`, rồi chạy [agent_metrics.py](../../starter/agent_metrics.py) với `--cases`, `--traces`, `--reviews`, `--split test`, `--output` để tổng hợp.
