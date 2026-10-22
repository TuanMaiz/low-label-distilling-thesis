# Context luận văn — cập nhật 19/10/2026

Tài liệu tổng hợp để đọc, sao chép và chuyển context sang cuộc trao đổi khác. Nguồn là ghi chép nghiên cứu, kế hoạch, artifacts hiện còn và các đính chính trong trao đổi với researcher. Đây là bản mô tả trạng thái, không phải hợp đồng thí nghiệm mới hoặc quyền chạy thêm.

## 1. Thesis hiện đang nghiên cứu điều gì?

Thesis nghiên cứu Entity Resolution/Entity Matching (ER/EM): quyết định hai records có cùng mô tả một thực thể hay không. Mục tiêu thực dụng là dùng LLM gán nhãn training pairs rồi huấn luyện compact cross-encoder để inference nhiều lần với chi phí thấp hơn direct LLM matching.

**Hướng nghiên cứu đã pivot sang uncertainty-aware LLM label refinement.** Thiết kế full-label gold versus LLM trên ba dataset và ba model là nền thực nghiệm; uncertainty không chỉ là một nhánh kỹ thuật nhỏ trong bản context.

Hai lớp câu hỏi nối tiếp nhau:

1. **Machine labeling:** Nhãn cứng do LLM sinh có thể thay thế thực dụng cho gold training labels khi train compact ER models, đồng thời giảm chi phí triển khai so với direct LLM matching hay không?
2. **Uncertainty-aware refinement:** Khi nào và bằng cách nào nên refine nhãn LLM để compact ER models học hiệu quả, khi quá trình phát hiện và xử lý không được dùng benchmark gold?

Các câu hỏi cụ thể của uncertainty pivot:

- Teacher labels sai hoặc thiếu đáng tin ở những tình huống nào?
- Teacher consistency, student uncertainty và teacher–student disagreement có nhận diện được nhãn đáng nghi không?
- Prune hoặc selective relabel có cải thiện downstream student không?
- Detector có nhầm những pair khó nhưng hợp lệ thành noisy labels không?
- Có những nhãn lỗi phát hiện được nhưng không ảnh hưởng đáng kể đến downstream không?
- Lợi ích có đáng với chi phí teacher passes, OOF, diagnosis, review và training bổ sung không?
- Quy tắc có chuyển dataset và student architecture được không?

Không mặc định refinement phải giúp. Kết quả refinement không cần thiết hoặc không đáng chi phí cũng hợp lệ.

Hướng trao đổi gần nhất là **tìm hard-but-useful pairs**: pair khó chưa chắc có nhãn sai; có thể student chưa học được khác biệt quan trọng và cần tăng attention/exposure. Weighting đang được tham khảo, chưa được chọn thành thuật toán hay arm chính thức.

## 2. Lịch sử hướng đi và các lần pivot

| Giai đoạn | Hướng | Trạng thái và điều đã ghi nhận |
|---|---|---|
| Ban đầu | Structured rationale distillation: teacher sinh giải thích có cấu trúc cho student | Đã dừng làm hướng chính; pilot khi đó kém label-only |
| Tháng 7 | Cost-aware, low-label LLM supervision; random versus active pair selection | Đã có pilot, sau đó rời phạm vi chính |
| Tháng 7 | Screening FLAN-T5, ModernBERT và Qwen | Qwen đủ tốt để đi tiếp; ModernBERT từng prediction collapse |
| Từ 20/8 | Full-label: 3 dataset × 3 compact model × gold/LLM labels | Là nền thiết kế thực nghiệm đang được ghi nhận |
| Tháng 9 | **Uncertainty-aware label-error detection và selective refinement** | Pivot nghiên cứu quan trọng; đã triển khai nhiều bước trên WDC |
| Đầu tháng 10 | Diagnosis-guided refinement: xem loại vấn đề trước khi chọn hành động | Đã có evidence-check và quick pilot |
| 14–15/10 | Instruction student nhấn mạnh product variants | Đã thử; không cải thiện F1 trong comparison lưu lại; rollback |
| 16–17/10 | Hard-but-useful pairs; similarity, weighting, RHO-LOSS | Đang tìm hiểu; chưa có thuật toán cuối hoặc run weighting |
| Trao đổi gần nhất | Vì sao student bỏ qua chi tiết định danh? | Hướng khám phá nhẹ, riêng; chưa có behavioral/mechanistic experiment |

### Những điều không được suy diễn từ lịch sử

- Rationale pilot thất bại không chứng minh mọi rationale distillation đều vô ích.
- Lý do đầy đủ của việc bỏ thiết kế low-label chưa được lịch sử xác nhận toàn bộ; không tự gán mọi thay đổi cho một kết quả đơn lẻ.
- Memo 10/9 ghi nhận uncertainty là hướng mở rộng từ giảng viên. Nội dung phản hồi chính thức đầy đủ và phạm vi cuối vẫn cần xác nhận.
- Ngày 18/9, phạm vi uncertainty được thu gọn còn **prune và relabel**, loại reweight để giảm số downstream runs. Weighting được mở lại trong trao đổi 16–17/10 như ý tưởng mới; chưa tự động trở thành arm được duyệt.
- Prompt variant của student thử tháng 10 khác với prompt variants từng được bàn để đo teacher consistency. Không nhập hai loại thí nghiệm này làm một.

### “LLM active” trong kết quả cũ là gì?

Đó là arm dùng **pair selection có chủ đích trước khi gọi LLM gán nhãn**, thay cho chọn ngẫu nhiên.

Selector `llm_active_bucketed_v1` dùng các đặc trưng như title/description similarity, model-code overlap, brand agreement/conflict, price/currency và missing fields. Nó chia candidates thành bốn nhóm:

1. Có vẻ dễ match.
2. Có vẻ khó match.
3. Có vẻ dễ non-match.
4. Có vẻ là hard negative.

Budget 128 mặc định lấy 32 pair mỗi nhóm. Đây là nhóm ứng viên theo heuristic, **không phải gold classes**. Gold không dùng để tính điểm chọn mẫu; sau selection, LLM mới gán nhãn.

“Active” ở đây là cách gọi lịch sử cho chọn dữ liệu một lần, không phải vòng active learning nhiều lượt. Workflow này đã được cho nghỉ khỏi luồng chính khi migration full-label ngày 20/8; code cũ giữ trong Git history.

## 3. Thiết kế nền và phạm vi chưa freeze

| Thành phần | Thiết kế đã ghi nhận |
|---|---|
| Dataset | Chính xác 3 benchmark |
| Compact models | Chính xác 3 model jointly encode cả hai records |
| Nguồn nhãn train nền | Benchmark gold và LLM hard labels |
| Core compact-model matrix | 18 cells |
| Direct LLM baseline | Một baseline mỗi dataset, tổng 3 |
| Run mỗi cell | Một run khai báo trước; không có chiều repeated-run |
| Metric chính | Match-class F1 |
| Metric bổ sung | Precision, recall, macro F1, accuracy, confusion counts, timing, throughput, cost, break-even |

Uncertainty pivot mở rộng nền này bằng teacher consistency, OOF, detection, diagnosis và treatments. **Ma trận cuối gồm những controls/ablations nào chưa freeze.** Không suy ra mọi detector và treatment phải chạy trên toàn bộ dataset × student matrix.

Đã chọn WDC, DBLP-ACM và Qwen3-Reranker-0.6B. Dataset thứ ba và hai compact models còn lại chưa chốt. Hợp đồng toàn cục vẫn draft. Các bước đã chạy được duyệt theo các hợp đồng/phase có phạm vi hẹp.

Ngoài phạm vi chính đã ghi: low-label budgets, active selection, rationale distillation, adaptive bi-/cross-encoder cascades, thêm dataset/model hoặc chiều repeated-run. Những ghi chép cũ về các hướng này không phải chỉ dẫn chạy tiếp.

## 4. Uncertainty pipeline và vai trò CL

Ba nguồn tín hiệu đã được nghiên cứu:

| Tín hiệu | Cách lấy | Giới hạn |
|---|---|---|
| Teacher consistency | Ba lượt labeling cùng teacher và prompt; xem agreement 3/3 hay 2/3 | Ba lượt cùng model có thể cùng sai |
| Student support/uncertainty | Xác suất OOF dành cho supplied teacher label; margin/entropy là các khả năng đã bàn | Student có thể phản đối một nhãn đúng vì chưa học được |
| Teacher–student disagreement | So teacher majority label với prediction OOF | Không xác định bên nào sai |

**CL = Confident Learning, triển khai bằng Cleanlab.** Nó thuộc bước detection/ranking, cần ghi riêng khỏi diagnosis.

- **Count:** ước lượng cấu trúc/khối lượng noise ở mức dataset và lớp, gồm confident joint.
- **Rank:** ưu tiên các pair đáng nghi từ probabilities và supplied labels.
- **Treatment:** quyết định prune, relabel hoặc retain; không tự có chỉ vì CL đã flag pair.

Luồng nghiên cứu dự kiến:

```text
Frozen training pairs
  → ba teacher passes
  → majority labels + consistency
  → ba-fold student OOF probabilities
  → CL / uncertainty / disagreement analysis
  → suspicious-pair selection (fusion chưa chốt)
  → evidence diagnosis
  → retain / prune / selective relabel
  → downstream student training
  → chất lượng và tổng chi phí
```

### Luồng đã chạy thực tế không được nhập với luồng dự kiến

- CL tạo **120 candidate pairs** và all-pair ranking.
- Evidence-check thực tế chọn **158 pair** từ union của **13 teacher 2/3** và **149 teacher–student disagreements**, trùng 4 pair.
- Không được gọi tập 158 này là tập 120 CL candidates. Quan hệ CL/consistency/disagreement chưa được đóng thành policy fusion cuối.
- Diagnosis là mô tả evidence/tình huống của records, không phải xác minh nhãn lỗi.
- Exact quote hợp lệ chỉ xác minh quote có trong nguồn, không đảm bảo diễn giải ngữ nghĩa đúng.

OOF hiện có một prediction held-out cho mỗi pair, không phải ba model votes độc lập trên mỗi pair. Chia theo pair chưa tự loại record hoặc product-family overlap giữa các fold.

### Ranh giới sử dụng gold

- Gold train dùng cho arm gold và đánh giá sau khi đóng băng operational outputs theo giao thức.
- Không dùng gold train/validation/test để tạo detector ranking, chọn pair hay quyết định sửa nhãn trong phương pháp được tuyên bố gold-free.
- Gold validation có vai trò checkpoint/decision-threshold selection đã quy định; không đồng nghĩa được dùng để học sample weights.
- Hidden training gold có thể được join sau khi outputs freeze để đo detection quality và phân tích teacher/detector errors.
- Test access theo từng authorization. WDC test về sau đã được chạy và xem; không áp câu “test chưa mở” của nhật ký cũ lên mọi thí nghiệm hiện tại.
- Gold-free không đồng nghĩa human-free/assistant-review-free.

### Các đối chứng từng được đề xuất, chưa phải thiết kế đã chốt

1. **Same-size random intervention:** ví dụ detector chọn 100 pair để prune, so với prune 100 pair ngẫu nhiên. Mục đích là xem chất lượng selection có thêm giá trị so với thay đổi cùng lượng dữ liệu hay không. Random relabel tương tự nhưng cần thêm API budget.
2. **Oracle prune:** dùng gold train để biết teacher label nào khác benchmark rồi loại các pair đó. Đây là đối chứng phân tích có đáp án, không phải phương pháp triển khai thiếu gold.
3. **Gold-supervised training:** arm train bằng gold labels. Arm này đã chạy cho WDC–Qwen; không nên gọi nó là trần F1 tuyệt đối.

Random/oracle controls có trong ghi chú và draft plan, nhưng **chưa có căn cứ coi researcher đã chốt tất cả hoặc các runs đó đã hoàn thành**. Oracle prune chưa chạy. Không tự thêm chúng vào workload được duyệt.

Detection quality (Precision@k, Recall@k, AUPRC/AP) và downstream quality là hai lớp evaluation được đề xuất; chưa có bộ kết quả hoàn chỉnh cho chúng.

## 5. Dữ liệu và model hiện có

| Dataset | Quy mô chính thức | Tình trạng |
|---|---|---|
| WDC Products, pair-wise 80% corner cases, small development, test 100% unseen products | Train 2.500; validation 2.500; test 4.500 | Đã có labels, targets, training, OOF, CL, diagnosis, refinement và một số test evaluations |
| DBLP-ACM, structured DeepMatcher snapshot | Train 7.417; validation 2.473; test 2.473 | Source verified và offline integration hoàn tất; chưa có production labels/model predictions |
| Dataset 3 | Chưa chọn | Chưa triển khai |

### WDC

- Dataset ID: `wdc_products_80cc_small_100un`.
- Train: 500 match, 2.000 non-match.
- Validation: 500 match, 2.000 non-match.
- Test: 500 match, 4.000 non-match.
- Dùng official splits, không tự resplit để làm đẹp kết quả.
- Test có 100% unseen products theo setting benchmark.
- Labels benchmark cũng có giới hạn; thấy tình huống trái trực giác chưa đủ xác nhận gold sai.
- Contamination của pretrained models/LLM với benchmark công khai chưa được loại trừ.

### DBLP-ACM

- Snapshot nội bộ: `deepmatcher-structured-dblp-acm-2018-06-29-a15b752f`.
- Official archive và independent HTTPS downloads byte-identical.
- Đã có frozen source/normalization/identity contract, strict adapter và atomic preparation.
- Train có 1.332 match/6.085 non-match; validation có 444 match/2.029 non-match.
- Đã chạy fake labeling 7.417 rows, không API calls, chi phí 0 USD; temporary integration đi qua target publisher và independent validator.
- Đã có Qwen execution profile/preflight/fixture verifier và command renderer.
- Chưa có real production labels, production targets, trained DBLP model, predictions hoặc normalized test JSONL.
- Official splits có record overlap; không tuyên bố entity-disjoint generalization.

### Compact models

- Đã chọn `Qwen/Qwen3-Reranker-0.6B`: jointly process hai records, lấy score từ logits `yes/no`, fine-tune bằng LoRA.
- Hai model còn lại chưa chọn.
- FLAN-T5 và ModernBERT là lịch sử screening; còn config không đồng nghĩa được chọn vào final matrix.
- Gemma từng được chuẩn bị tooling; không có kết quả chất lượng được xác nhận trong hồ sơ đã đọc.
- Các backend đã có: seq2seq, classification và generative reranker.

## 6. Các thành phần hệ thống

| Khối | Thành phần hiện hữu | Giới hạn |
|---|---|---|
| Data | Schema, loaders, serialization, source checks, split/identity/hash/overlap checks | Chưa đủ ba dataset |
| LLM labeling | OpenRouter client, structured outputs, provider pinning, attempts/audit, reconciliation và cost accounting | Production đầy đủ hiện ở WDC |
| Target publication | Builder và independent validator; fail khi thiếu/trùng/thừa/sai identity | WDC targets thật; DBLP mới fake/temporary integration |
| Student | Config-driven train/evaluate; Qwen LoRA, adapters/merged models, threshold selection | Chưa hoàn tất ba architectures |
| OOF | Deterministic fold preparation, verifier, GPU runner/recovery và exact-once merge | Pair-level OOF |
| CL | Cleanlab 2.9.0, confident joints, noise estimate, all-pair scores và candidates | Candidates chưa là verified errors |
| Evidence diagnosis | Blinded record-only requests, bảy case definitions, quotes, reports và cost | Model-generated observations |
| Refinement | Quick dataset builder và explicit decision ledger | Review-assisted pilot, chưa là final automatic policy |
| Reproducibility | Artifact contracts, checkpoint manifests, runtime provenance, checksums, package/recovery checks | Một số artifacts lịch sử thiếu |
| Analysis | Metrics/cost utilities, FP/FN reports, full input và training analogues | Chưa có thesis-wide aggregate |
| Writing/context | Plans, journals, advisor briefs, decisions/ideas/techniques/open questions | Writing plan cần đổi framing |

Các file lõi gồm `data/schema.py`, `data/er_dataset_loader.py`, `data/serialize_pairs.py`, `supervision/build_teacher_consensus.py`, `supervision/prepare_oof_folds.py`, `supervision/build_confident_learning.py`, `supervision/run_evidence_check.py`, `supervision/build_quick_refined_dataset.py`, `experiments/train_student.py`, `experiments/evaluate_student.py`, `models/generative_reranker_student.py` và các utilities provenance/metrics/cost.

Engineering checks và independent artifact validation không tự chứng minh phương pháp cải thiện F1. Không có training/API call/test suite mới được chạy để viết bản context này.

## 7. Kết quả thực nghiệm đến hiện tại

### 7.1. Rationale và low-label screening lịch sử

Các dòng dưới thuộc thiết kế cũ và là validation results; không trộn thành final uncertainty results.

| Thử nghiệm | Match F1 |
|---|---:|
| Label-only trong rationale pilot ban đầu, 128 training examples | 0,4686 |
| Structured rationale, 122 examples có rationale hợp lệ | 0,3639 |
| FLAN-T5 gold random, budget 128 | 0,5251 |
| FLAN-T5 LLM random | 0,5441 |
| FLAN-T5 LLM active bucketed | 0,5345 |
| Qwen gold random, budget 128 | 0,7104 |
| Qwen LLM random | 0,7267 |
| Qwen LLM active bucketed | 0,7472 |

Rationale tăng recall nhưng giảm precision trong pilot đó; số training examples cũng không hoàn toàn bằng nhau. Không gọi đây là matched comparison hoàn hảo hay kết luận rationale nói chung vô ích.

Qwen active có signal tốt hơn random ở run quan sát, nhưng chưa chứng minh ưu thế ổn định. First ModernBERT diagnostic bị collapse gần một lớp ở một số arms; chưa là phán quyết công bằng về kiến trúc.

### 7.2. Direct LLM lịch sử: làm rõ nguồn 0,8734

**Đã xác minh từ artifact còn trong Git:** đây là lượt gọi API trả phí `openrouter:openai/gpt-5.4-mini` của dự án trên **2.500 WDC validation pairs**, không phải official test split.

| Metric | Giá trị |
|---|---:|
| Match precision | 0,870775 |
| Match recall | 0,876000 |
| Match F1 | **0,8733798604**, thường làm tròn 0,8734 |
| Macro F1 | 0,920803 |
| Accuracy | 0,9492 |
| TP / FP / TN / FN | 438 / 65 / 1.935 / 62 |
| Valid / invalid | 2.500 / 0 |
| Token-accounted estimated cost | **0,7964985 USD** |

Historical cost artifact ghi `split=validation`, input là serialized validation, 969.364 input tokens và 15.439 output tokens. Resume tái dùng 386 valid rows và viết thêm 2.114 rows. Đây là evidence của một lượt “test thử model trả phí” nếu dùng từ test theo nghĩa thông thường; **không đồng nghĩa test split chính thức**.

Hai files predictions/cost hiện đang bị xóa trong working tree nhưng còn ở Git HEAD. Bản context chỉ đọc lịch sử, không restore hoặc thay đổi chúng. Chi phí là số ước tính ghi trong artifact, không phải invoice được đối soát độc lập.

Researcher cũng nhớ đã có một mốc tốt nhất lấy từ paper/test, nhưng chưa nhớ paper/model. **Chưa xác minh được mốc đó có phải con số khác hay không.** Giữ nó như một literature reference chưa truy lại nguồn; không gán 0,8734 cho paper/test, không gọi là SOTA, không dùng làm matched comparison với test hiện tại.

### 7.3. Screening labeler trên 300 WDC training pairs

Một sample uniform random seed 42, giữ natural class prevalence; gold chỉ cho comparison, không gửi labeler. OpenRouter pinned OpenAI upstream, fallbacks disabled.

| Setting | Đúng/tổng | Match F1 | Received-response cost |
|---|---:|---:|---:|
| Sol-high | 291/300 | 0,921739 | 0,327135 USD |
| Sol-max | 291/300 | 0,920354 | 0,386535 USD |
| Sol-pro-max | 292/300 | 0,929825 | 2,7461713 USD |

**Sol-high được chọn vì cân bằng chất lượng và chi phí.** Sol-pro-max đúng hơn một pair trên sample; không nói Sol-high luôn chính xác nhất.

### 7.4. Full-label WDC và single-pass gold–LLM vertical slice

- Đã có 2.500/2.500 valid Sol-high labels.
- Tái dùng 300 screening labels, gọi thêm 2.200 training pairs.
- Cumulative labeler cost: **2,693225 USD**, dưới ceiling 5 USD.
- Gold–LLM agreement: **2.421/2.500 = 96,84%**, có 79 disagreements.
- Gold và LLM targets có byte-identical ordered pair IDs/input text; target source khác nhau.

| Qwen training labels | Validation match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|
| Gold | **0,820840951** | 0,885330870 | 0,9216 |
| Single-pass LLM-hard | **0,814126394** | 0,881578997 | 0,9200 |

LLM-hard thấp hơn 0,006715 match F1, khoảng 0,67 điểm phần trăm. Một run mỗi arm trên RTX 3090, bf16. Training lần lượt khoảng 1,70 và 1,89 GPU-hours; inference khoảng 33–34 pairs/s trong runtime đó. Hai arm có 2.500 valid validation predictions, zero invalid.

Đây là validation-only comparison, không phải gold–LLM final-test comparison. Nếu uncertainty main study dùng majority labels thì single-pass result này là pilot/engineering evidence, không tự trở thành final majority cell.

### 7.5. Ba teacher passes và majority labels

| Pass | Valid labels | Cost |
|---|---:|---:|
| Pass 1 | 2.500 | 2,693225 USD |
| Pass 2, fresh | 2.500 | 2,709705 USD |
| Pass 3, fresh | 2.500 | 2,693205 USD measured valid-response cost |

Tổng các cost trên là **8,096135 USD**, không phải toàn bộ thesis cost. Pass 3 có một transient failed attempt rồi retry thành công; không suy ra mọi cost ngoài valid responses đều đã reconcile.

Majority bundle:

- 2.500 labels: **495 match, 2.005 non-match**.
- **2.487 pair 3/3** và **13 pair 2/3**.
- Pass mới không tái dùng screening cache làm judgment mới.

Consistency có ít phân giải trong bộ này vì gần hết unanimous; 3/3 không chứng minh đúng.

### 7.6. OOF và CL

- Hoàn tất ba-fold Qwen OOF trên Tesla V100-SXM2-32GB.
- Fold 1: 1.333 train / 333 inner validation / 834 held-out.
- Fold 2 và 3: mỗi fold 1.334 / 333 / 833.
- Merge có **2.500 unique OOF probabilities**, checksum và independent rederivation pass.
- Cleanlab 2.9.0 tạo **120 candidates (4,8%)**.
- Calibrated error-count estimate: **124 (4,96%)**.
- Có raw/calibrated confident joints, estimated joint distribution, all-pair scores và candidate IDs.

Đây là model estimates; chưa phải noise rate được gold xác nhận. CL và teacher–student disagreement cùng lấy thông tin từ OOF, không được tính như hai xác nhận độc lập.

### 7.7. Evidence-check và quick-refinement

Evidence-check selected set: **13 + 149 − 4 = 158 unique training pairs**.

- Hiện **155/158 valid diagnoses**; còn ba literal-quote failures.
- Bảy definitions: representation difference, one-sided information, cross-record conflict, ambiguous variant/bundle, intra-record contradiction, ambiguous role/scope, noisy/malformed data.
- Counts trong 155 valid diagnoses lần lượt là 99, 138, 62, 105, 7, 46, 103. Multi-label nên không cộng thành 155; sample được chọn có chủ đích nên không phải whole-dataset prevalence.
- Known received-response cost: **3,253112 USD**.
- Ba historical dispatches chưa reconcile billing hoàn toàn; các pair đó đã có valid new diagnoses. Không suy ra diagnosis coverage từ unresolved billing IDs.

Quick-refinement là offline, dùng evidence sẵn có và 65 explicit assistant-reviewed decisions:

| Hành động | Số pair |
|---|---:|
| Giữ nhãn | 2.492 |
| Prune | 7 |
| Relabel match → non-match | 1 |
| Retained train | **2.493** |

Retained classes: 489 match, 2.004 non-match. Pair relabel là `88031640#6256141`, listed RAM 8 GB versus 16 GB. Pilot hard-prune valid case-5 (intra-record contradiction) trước; difficult/unanalysed/invalid cases giữ majority labels.

Đây là **assistant-reviewed experimental dataset**, không phải corrected ground truth. Không phải toàn bộ semantic decisions được thực hiện bằng một automatic detector; policy prune case 5 cũng chưa được chứng minh tối ưu.

### 7.8. Downstream majority/refined và prompt variant

| Lượt/điều kiện | Validation match F1 | Test match F1 |
|---|---:|---:|
| Majority-unrefined, metrics lưu riêng | **0,819119025** | Chưa có trong artifacts hiện đọc được |
| Quick-refined, metrics lưu riêng | **0,801099908** | — |
| Quick-refined prompt gốc, gói có test về sau | **0,811374408** | **0,758122744** |
| Quick-refined variant-aware instruction | **0,809174312** | **0,746598639** |

Các dòng quick-refined là các lượt khác nhau. Không nhập metrics hoặc thresholds của chúng thành một run. Table dùng final evaluation metrics; không thay bằng best epoch metrics trong training summary vì các số đó có thể khác.

Trong comparison validation lưu riêng, quick-refined thấp hơn majority. Hiện chưa có bằng chứng refinement đã giúp downstream; cũng thiếu majority-baseline test tương ứng để xác định hiệu quả refinement trên test.

Test comparison prompt gốc/variant:

| Metric | Prompt gốc | Variant |
|---|---:|---:|
| Precision | 0,690789 | 0,649408 |
| Recall | 0,840000 | 0,878000 |
| TP / FP / TN / FN | 420 / 188 / 3.812 / 80 | 439 / 237 / 3.763 / 61 |
| Validation-selected threshold | 0,061875984 | 0,002234363 |
| Test max input length | 4.096 | 4.160 |

Variant test tăng max length vì có input vượt 4.096; không nói mọi setting đều giống tuyệt đối. Một run không xác định nguyên nhân, không chứng minh shorter prompts nói chung tốt hơn.

Variant bị researcher yêu cầu rollback ngày 15/10. Instruction mặc định tiếp theo:

> Determine whether Record A and Record B describe the same real-world product. Answer yes only when they refer to the same product.

Config/checkpoint variant giữ nguyên để reproduce; không đổi instruction của variant-trained checkpoint rồi coi đó là tái lập cùng điều kiện.

## 8. Điều đã học từ error analysis

Quick-refined prompt gốc có validation F1 0,8114 và test F1 0,7581. Chênh lệch không tự chứng minh model reject non-match kém hơn:

- Match prevalence: validation 20%, test 11,11%.
- False-positive rate: **6,35% → 4,70%**.
- Recall: **85,6% → 84,0%**.

Thay prevalence để minh họa số học không tạo ra corrected score; published test F1 vẫn 0,7581.

Đã quan sát các FP với khác flavor, pack count, model suffix hoặc accessory versus main product; có FN dù hai phía có shared model codes. Một số pair thiếu evidence hoặc có căng thẳng giữa visible semantics và benchmark identity policy. Các nhận xét này chưa xác minh gold sai.

Có 46 FP score ≥ 0,9, trong đó 15 FP ≥ 0,99. Vì vậy không phải mọi lỗi chỉ nằm sát threshold; score cao cũng chưa tách được calibration, ranking/discrimination và supervision effects.

Đã có báo cáo 30 test errors (18 FP, 12 FN) với full inputs và training analogues. Đây là purposive diagnostic sample, không phải prevalence sample hoặc causal proof. Một số artifacts chỉ còn manifest references, chưa chứng minh byte-level GPU training inputs đã mất.

**WDC test đã được xem và ảnh hưởng phương pháp/prompt ideas về sau.** Không gọi nó là pristine confirmation cho các phương pháp được hình thành sau khi xem lỗi. Cần dữ liệu chưa dùng để thiết kế để kiểm tra transfer.

## 9. Hướng đang tìm hiểu: hard-but-useful pairs

Phải tách ba tính chất:

1. Pair khó đối với model.
2. Supplied label đáng tin.
3. Học pair đó hữu ích cho downstream.

Không phương pháp đang bàn nào tự chứng nhận cả ba. Researcher muốn tìm cách hệ thống, tái dùng được ở dataset sau, để student học khác biệt identity nhỏ; đồng thời lo rằng thêm LLM judgments chỉ thêm một bước chưa đáng tin.

### Các ý tưởng gần nhất

- **Similarity/clustering:** tìm candidates có vẻ giống nhưng non-match hoặc khác biểu diễn nhưng match. Chưa chọn embedding, metric, cutoff hay label-check policy. Similarity không xác minh identity; code-distance không là semantic oracle.
- **JTT:** tăng exposure/weight cho tập model ban đầu làm sai. Nhãn teacher sai có thể cũng nằm trong tập này. Dùng OOF errors thay early training errors là adaptation.
- **Dataset Cartography:** theo dõi supplied-label confidence/variability qua epoch. Chưa có per-pair trajectories; final OOF scores không thay được learning history.
- **RHO-LOSS-inspired reference signal:** pair current student còn khó nhưng model không train trên pair đó vẫn ủng hộ supplied label có thể đáng tăng chú ý.

### RHO-LOSS đã bàn đến đâu?

Researcher đã yêu cầu giải thích và tải paper để đọc. PDF hiện có ở `documents/rho-loss-2022.pdf`. Chưa chọn triển khai.

Khả năng adaptation đã được assistant đề xuất:

```text
reference_loss = -log(P_OOF của supplied label)
priority = current_per_pair_loss - reference_loss
```

Với match dùng match probability; với non-match dùng non-match probability. OOF hiện có đủ để tính reference loss. Current per-pair loss vẫn phải lấy trong target training; chỉ xếp hạng từ OOF loss là phương pháp khác.

RHO-LOSS gốc chọn batches; capped bonus weighting là adaptation chưa chốt. Cap, normalization, update schedule, class balance và loss-scale comparability chưa freeze.

Rủi ro cần kiểm tra:

- Reference cũng bỏ qua đúng chi tiết identity cần học thì pair có thể không được ưu tiên.
- Reference và target cùng học quy tắc sai có thể khuếch đại teacher errors.
- Loss scales khác giữa fold và training stage.
- Weighting làm lệch class balance hoặc để vài families chiếm ưu thế.
- Existing OOF dùng majority labels; quick-refined là bundle khác, không silently trộn provenance.
- Dataset sau cần reference predictions và thêm compute cost.

Audit train-only reference support trước GPU experiment là đề nghị trong memo, chưa phải việc đã thực hiện. Weighting chưa có run hoặc demonstrated gain.

## 10. Hướng riêng: vì sao model bỏ qua chi tiết?

Researcher muốn khám phá nhẹ câu hỏi vì sao model hành xử như vậy, không chỉ thử thêm cách cứu F1.

Những khả năng đã nêu, chưa phải nguyên nhân xác nhận:

- Model dựa mạnh vào shared text hoặc product-family similarity.
- Supervision chưa dạy được subtle identity distinctions.
- Chi tiết có thể bị ảnh hưởng bởi serialization/tokenization/input coverage.
- Score có đổi nhưng chưa vượt fixed threshold.
- Visible evidence không đủ hoặc identity policy của benchmark khác trực giác.

Hướng kiểm tra đã bàn:

- Theo CheckList: controlled changes vào identity attributes và các thay đổi vô hại, đo cả score lẫn decision.
- Thay lượng shared text nhưng giữ identity evidence để xem có yếu tố đóng góp không.
- Xem cả flagged/unflagged và correct/error examples; không chỉ kể vài ví dụ phù hợp giả thuyết.
- Mechanistic interpretability qua activations/interventions là lựa chọn sâu hơn, chưa chốt làm.

Chưa có behavioral suite hoặc mechanistic experiment. Attention maps/raw weights hay một lần đổi prediction không đủ chứng minh cơ chế nhân quả.

## 11. Literature và trạng thái tìm hiểu

Có PDF hoặc memo không đồng nghĩa đã đọc kỹ toàn văn. Không gán “đã đọc xong” nếu hồ sơ không xác nhận.

| Nhóm | Paper/phương pháp | Vai trò và trạng thái |
|---|---|---|
| Gần nhất với hướng labeling | Steiner & Bizer, *Labeling Training Data for Entity Matching Using LLMs* | Baseline và novelty boundary quan trọng; đã có analysis memo |
| Distillation/explanations | DistillER; Wadhwa et al. | Ranh giới của rationale/knowledge-distillation claims; PDFs có sẵn |
| Direct LLM matching | Peeters/Steiner/Bizer; fine-tuning EM work | Background và literature context; không tự coi scores khác setting là matched baseline |
| Annotation consistency | Pangakis & Wolken | Nền cho repeated labeling; consistency signal không bảo đảm filtering giúp downstream |
| Detection | Confident Learning; Annotation Error Detection survey | CL đã triển khai; survey làm khung các phương pháp/evaluation |
| Training dynamics | Dataset Cartography; AUM | Đang tham khảo, chưa có logging tương ứng |
| Useful hard examples | RHO-LOSS; JTT | RHO-LOSS được yêu cầu tải để đọc; chưa triển khai |
| Noisy-label learning | ELR; UA-MT; RDA; PASS; DivideMix; Co-teaching+ | Tham khảo, chưa chọn làm pipeline |
| Weight learning | Learning to Reweight Examples | Clean validation tham gia optimization; không khớp vai trò gold validation hiện tại nếu dùng nguyên bản |
| Diagnosis/repair | DeMix; error decomposition | Tham khảo cách detect–diagnose–repair; original supervision assumptions không tự chuyển sang gold-free ER |
| Analysis methodology | Errudite; CheckList | Đã có reading notes/principles; chưa triển khai behavioral suite |
| Slice discovery | Domino; SliceLine; Spotlight; human evaluation of slice discovery | Tham khảo; chưa đưa detector vào operational pipeline |
| Confidence/risk | Calibration; selective classification; conformal risk control | Chưa triển khai; abstention đổi mục tiêu full-coverage |
| ER hard-pair precedents | Block-SCL; Sudowoodo; IDP-EM | Related work; IDP-EM mới kiểm tra publisher summary, chưa xác minh full equations/ablations |

Những giới hạn đã bàn:

- UA-MT dùng EMA teacher/consistency masking, khác external LLM teacher.
- RDA cho phép cả hai binary labels có thể mất supervised signal của pair; khác soft label 0,5/0,5.
- PASS cần peer-training setup; ba OOF fold không phải ba held-out peers cho mỗi pair.
- ELR cần bằng chứng memorization; final OOF không cho training trajectory.
- Semantic entropy cho nhãn binary chuẩn hóa không tự thêm nhiều ngoài vote count, không phát hiện systematic unanimous errors.
- Soft labels/multi-output đổi hard-label contract.
- Positive-temperature scaling không tự cải thiện attainable ranking.
- Domino/slice discovery gốc có task-label/reference requirements; teacher disagreement slices chưa là verified error slices.

**Novelty chưa được xác lập.** Hard-example weighting, similarity mining và label refinement đều có tiền lệ. Combination được đề xuất cần full-text comparison và empirical controls; không coi memo đề xuất novelty là kết luận novelty đã chứng minh.

## 12. Những quyết định cần giữ khi tiếp tục

- Giữ official splits và dataset-version identities; không sửa test labels hoặc bỏ test errors để tăng metric.
- Gold không vào labeler prompt hoặc detector/treatment decisions trong gold-free method.
- Teacher hiện dùng GPT-5.6 Sol-high; OpenRouter pin OpenAI upstream, không fallback tùy ý.
- Teacher labels, majority, CL scores, diagnoses và refined targets là artifacts riêng; không mutate original input JSONL để đính mọi field.
- Student instruction mặc định đã quay về bản gốc; giữ config/checkpoint variant để reproduce đúng lịch sử.
- Current decision-threshold rule tối ưu validation macro F1; tie-break match F1, accuracy, gần 0,5. Saved threshold dùng nguyên trên test. Chưa chốt đổi objective hoặc floor 0,3/0,5.
- Match F1 vẫn là metric nghiên cứu chính; khác biệt với threshold objective cần được giải quyết rõ trong contract.
- Không chọn thêm runs hoặc thuật toán chỉ vì WDC test score đẹp hơn.
- Không tự biến literature recommendations, random/oracle controls, reweighting ideas hoặc hướng “vì sao” thành execution authorization.
- Cost analysis phải tính labeling, extra passes, OOF, diagnosis/relabel, training và inference. GPU-rate scenarios khác API usage cost; benchmark-provided gold không đồng nghĩa human annotation cost bằng 0.
- Break-even nền: `N* = (C_label + C_train) / (c_direct - c_model)`, bổ sung preparation/refinement costs cho phương pháp cuối. Không có finite break-even nếu mẫu số không dương.

## 13. Việc chưa chốt và bằng chứng còn thiếu

1. Dataset 3 và models 2–3; scientific contract cuối và yêu cầu mở rộng chính thức từ giảng viên.
2. Refinement/weighting/nhánh “vì sao” sẽ có vai trò nào trong luận văn.
3. Fusion consistency–CL–disagreement, ranking definition, selection budget và treatment policy.
4. Khả năng detector bỏ sót unanimous teacher errors và bias do pair-level OOF record/family overlap.
5. Evaluation detection bằng hidden gold sau freeze; random/oracle controls nào thực sự cần và được duyệt.
6. Matched downstream comparison, đặc biệt majority-baseline test còn thiếu trong artifacts hiện có.
7. Threshold objective, calibration và generalization; chưa rõ nguồn gốc quyết định tối ưu macro F1 thay vì match F1.
8. Reference support/loss scales cho RHO-style adaptation; dynamic current losses và training histories chưa có.
9. Ba literal-quote failures và billing reconciliation historical dispatches; chưa có quyền gọi thêm từ vòng đã kết thúc.
10. Một số outputs/weights/training inputs lịch sử thiếu; phân biệt provenance mạnh/yếu của từng result.
11. Literature baseline researcher nhớ: paper/model/test setting chưa truy lại; không trộn với verified paid validation result 0,8734.
12. DBLP chưa có production labels/model results; chưa có transfer evidence sang dataset/model khác.
13. Thesis-wide cost/break-even, direct-LLM baselines đúng scope, contamination và license/attribution assessment còn thiếu.
14. Writing plan còn framing low-label cũ; cần cập nhật trước khi viết các chương theo hướng uncertainty hiện tại.

## 14. Hồ sơ nguồn để tiếp tục

Các đường dẫn dưới tương đối với repository root; có thể tìm bằng tên file khi chuyển context sang môi trường khác.

### Discussion records

- `docs/research/decisions.md`
- `docs/research/ideas.md`
- `docs/research/techniques.md`
- `docs/research/open-questions.md`

### Plans và pivot

- `plans/260820-1507-full-label-er-migration/plan.md`
- `plans/260820-1507-full-label-er-migration/research/experiment-contract.md`
- `plans/260820-1507-full-label-er-migration/research/260910-uncertainty-aware-label-refinement-literature.md`
- `plans/journals/2026-09-16-confident-learning-roles-in-uncertainty-aware-label-refinement.md` — có scope update ngày 18/9.
- `plans/journals/2026-09-28-post-hoc-error-analysis-for-llm-label-refinement.md` — random/oracle là proposed controls.
- `plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/plan.md`
- `plans/261005-0916-wdc-teacher-evidence-case-classification/plan.md`
- `plans/261006-0927-wdc-quick-evidence-refinement/plan.md`

### Results và provenance

- `journal/260707-phase-03-bucketed-active-selection.md`
- `journal/260710-phase-03-direct-llm-baseline-refresh.md`
- `outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.cost.json` — deleted working-tree file, đọc được từ Git HEAD; predictions tương ứng cũng còn trong Git.
- `docs/journals/260719-2158-phase05-experiment-results.md`
- `docs/journals/260724-1537-qwen-screening-phase6-decision.md`
- `labeller-screening/artifacts/comparison/comparison.csv`
- `labeller-screening/artifacts/predictions/*.audit.jsonl` — source của screening usage costs.
- `plans/260820-1507-full-label-er-migration/execution/260827-1612-wdc-qwen-gold-llm-hard/reports/260831-verified-validation-results.md`
- `outputs/uncertainty_refinement/wdc-qwen-oof/oof_summary.json`
- `outputs/uncertainty_refinement/wdc-qwen-cl/cl_summary.json`
- `outputs/uncertainty_refinement/wdc-evidence-check/summary.json`
- `data/cache/wdc_products/refinement/quick-tier-v1/summary.json`
- `outputs/majority-baseline/validation-metrics.json`
- `outputs/quick-refine/validation-metrics.json`
- `outputs/test-rerun/quick-refined-results.tar.gz` — separate later run, validation/test metrics.
- `outputs/wdc-qwen-quick-refined-variant-identity-v1-predictions-metrics.zip`
- `outputs/analysis/wdc-quick-refined-test-2026-10-13/report.md`
- `outputs/analysis/wdc-test-vs-three-pass-30-2026-10-13/summary.json`
- `documents/261014-student-variant-instruction.md`

### Reading và writing

- `plans/260820-1507-full-label-er-migration/research/260914-method-to-paper-map.md`
- `plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md`
- `plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md`
- `documents/rho-loss-2022.pdf`
- `documents/260903-thesis-approval-brief/thesis-approval-brief.md`
- `plans/260704-distiller-wdc-thesis-writing/plan.md` — legacy framing cần sửa.

Một số overview files có phase status cũ hoặc tên workflow lịch sử. Khi có mâu thuẫn, đối chiếu execution updates, phase checklists, result artifacts và các quyết định mới hơn; không lấy riêng frontmatter hoặc câu “test locked” cũ làm trạng thái toàn bộ project.
