# Câu hỏi còn mở và việc cần kiểm chứng

Ngày tổng hợp: 2026-10-19. Các mục dưới đây được rút từ trao đổi và ghi chú đã có; không phải task mới được cấp quyền thực hiện. `[CẦN XÁC NHẬN]` đánh dấu chỗ lịch sử chưa đủ chắc hoặc chưa chốt, không được tự điền câu trả lời.

## Q01 — Phạm vi cuối cùng và đóng góp khoa học

- Chính xác dataset thứ ba và hai compact model còn lại là gì? Hợp đồng chung còn mục nào chưa freeze?
- Refinement, weighting hay câu hỏi “vì sao” sẽ đóng vai trò nào trong luận văn? Kế hoạch cũ chưa có reweight, còn hướng “vì sao” được researcher mô tả là khám phá riêng.
- Researcher nói giảng viên đánh giá so sánh ban đầu còn đơn giản nên cần mở rộng; chưa có nội dung phản hồi đầy đủ để suy ra yêu cầu mới cụ thể. `[CẦN XÁC NHẬN]` yêu cầu chính thức của giảng viên và thiết kế cuối sau mở rộng.
- Hard-example weighting, uncertainty weighting và similarity mining đã có tiền lệ. Điểm khác biệt của đề xuất hiện tại là gì, và có vượt được baseline đơn giản không? Chưa có bằng chứng novelty.
- Nguồn: [kế hoạch chính](../../plans/260820-1507-full-label-er-migration/plan.md), [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md), trao đổi trực tiếp.

## Q02 — Mức F1 khoảng 0,75 nói được điều gì?

- Kết quả nào đang được gọi là “baseline”: single-pass, majority-unrefined hay quick-refined? Không dùng checkpoint single-pass làm đối chứng như thể cùng phạm vi.
- Đã có comparison majority-unrefined versus quick-refined đủ tương đương để đánh giá riêng hiệu quả refinement chưa? Các trao đổi hiện có chưa xác lập đầy đủ comparison đó.
- Test F1 0.7581 so với variant 0.7466 cho thấy variant run không cải thiện; nguyên nhân cụ thể của thay đổi chưa được xác định. Không thể từ một con số kết luận mọi giả thuyết về teacher labels/hard pairs thất bại.
- Các checkpoint/predictions baseline còn giữ đủ gì để phân tích mà không train lại? Researcher từng nói một số outputs đã mất; phạm vi còn thiếu hiện tại `[CẦN XÁC NHẬN]`.
- Nguồn: Trao đổi trực tiếp; [hồ sơ prompt](../../documents/261014-student-variant-instruction.md).

## Q03 — Đâu là lỗi dữ liệu, lỗi teacher và lỗi student?

- Với những pair có chi tiết khác biệt nhỏ, dataset có định nghĩa identity ở mức variant nào? Color/bundle/accessory có được so đúng vai trò và bộ phận không?
- Có bao nhiêu nhãn teacher thực sự xử lý đúng khác biệt đó? Researcher đã nhắc rằng xem vài pair chưa đủ xác nhận teacher nói chung làm được.
- Record thiếu information, nhãn sai và student chưa học tốt có thể gây cùng kiểu disagreement. Cần bằng chứng nào để phân biệt mà không dùng gold làm oracle trong detector?
- Pair `49982462#45065236` có đủ evidence identity trong toàn bộ input không? Có được dùng làm động cơ cho một quy tắc training cụ thể hay chỉ là nhận xét khi xem test? `[CẦN XÁC NHẬN]`.
- Nguồn: Trao đổi xem báo cáo; [nhật ký review](../../plans/journals/2026-10-04-wdc-first-five-pair-researcher-review.md).

## Q04 — Khi nào prune, relabel, giữ hoặc upweight?

- Rule nào định nghĩa “không thể quyết định từ evidence” mà không xóa các mẫu hợp lệ nhưng khó?
- Chẩn đoán có nhiều case chồng lấp thì thứ tự xử lý cụ thể ra sao? Within-tier order chưa được chốt đầy đủ.
- Pilot prune case 5 có loại cả mâu thuẫn family-level chưa thực sự ảnh hưởng specific identity không? Chưa chứng minh policy đó tối ưu.
- Khi sửa nhãn và tăng weight cùng lúc, làm sao biết yếu tố nào tạo thay đổi downstream? Assistant đề nghị tách hai can thiệp; chưa có giao thức được chốt.
- Manual/assistant review có vai trò nào trong development và execution? Không được gọi human-assisted policy là hoàn toàn tự động hoặc gold-free đồng nghĩa human-free.
- Nguồn: [nhật ký xử lý evidence và pilot](../../plans/journals/2026-10-06-priority-ordered-evidence-handling.md), [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md).

## Q05 — Detector có bỏ sót lỗi nhất quán không?

- Teacher 3/3 vẫn có thể cùng sai. Cách nào khảo sát các pair không bị flag thay vì chỉ kiểm tra union đáng nghi?
- Gộp teacher consistency, CL và teacher–student disagreement bằng quy tắc nào? Chưa chốt fusion.
- CL và disagreement cùng xuất phát từ OOF; cải thiện có thực sự thêm thông tin hay đếm lại cùng tín hiệu?
- Pair-level OOF cho phép record/family xuất hiện ở train fold khác. Điều này ảnh hưởng bao nhiêu đến usefulness của reference? Chưa được đo trong trao đổi này.
- Nguồn: [nhật ký CL](../../plans/journals/2026-10-02-wdc-cleanlab-oof-diagnostics.md), [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md).

## Q06 — Similarity có tìm được đúng hard-but-useful pairs không?

- Metric/embedding nào vừa nhận ra quan hệ ngữ nghĩa vừa không bỏ qua chi tiết identity nhỏ?
- Similarity cao có đang chủ yếu lấy true matches/easy negatives hoặc shared boilerplate? Có bỏ sót hard positives do khác ngôn ngữ, thiếu trường hoặc cách mô tả không?
- Có bị chi phối bởi một vài product families, độ dài hoặc category không? Cutoff có chuyển dataset được không?
- Sau mining, kiểm tra nhãn bằng cách nào đủ đáng tin? Researcher lo thêm LLM judgments chỉ thêm một bước chưa đáng tin; vote ba lần không giải quyết sai có hệ thống.
- Nguồn: Trao đổi trực tiếp; [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md).

## Q07 — OOF hiện có có dùng tốt cho RHO-LOSS adaptation không?

- Reference có thực sự ủng hộ đủ các training pairs chứa khác biệt nhỏ cần dạy không? Nếu cả reference lẫn target đều bỏ qua chi tiết, priority có thể thấp.
- Loss scale có tương đương giữa fold, giữa lớp và với target trong từng giai đoạn training không?
- Dùng đúng bundle majority labels như OOF, hay muốn làm trên refined bundle? Không silently trộn hai điều kiện.
- Cần lấy current per-pair loss trong training như thế nào? Saved OOF file chỉ cung cấp reference; chưa có dynamic priority hoàn chỉnh.
- Chọn batch như nguyên bản hay bounded weighting như assistant đề xuất? Cap, normalization, lịch cập nhật và tác động class balance chưa freeze.
- WDC có thể tái dùng reference scores; ở dataset sau, thêm reference training cost có còn đáng so với direct LLM không?
- Nguồn: [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md), trao đổi tái dùng OOF.

## Q08 — Student có ghi nhớ nhãn lỗi về cuối training không?

- Có thực sự xuất hiện early learning rồi memorization, hay mẫu khó cần nhiều thời gian để học đúng?
- Có đủ per-pair history/checkpoint để phân biệt không? Final OOF probabilities không cung cấp trajectory.
- Nếu bổ sung logging trong run tương lai, Cartography hoặc ELR có cho thông tin thêm không? Đây mới là đề xuất, chưa có kết quả.
- Nguồn: [memo uncertainty](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md).

## Q09 — Vì sao decision threshold rất thấp?

- Threshold thấp đến từ phân bố score, calibration, quá trình training hay yếu tố nào khác? Hiện chỉ biết đó là nghiệm tối ưu validation theo rule đang dùng, chưa có giải thích nguyên nhân.
- Tối ưu macro F1 trong khi metric chính là match F1 có phù hợp hợp đồng nghiên cứu không? Ai/ngày nào chốt objective này `[CẦN XÁC NHẬN]`.
- Ngưỡng có ổn định và chuyển sang dữ liệu khác tốt không? Validation optimization có thể overfit; không dùng test để đổi rule rồi trình bày như xác nhận mới.
- High-score FP là ranking/discrimination failure, miscalibration, threshold effect hay sự kết hợp? Một score cao không đủ phân định.
- Nguồn: Trao đổi trực tiếp; [threshold selector](../../utils/classification_threshold.py).

## Q10 — “Tại sao model bỏ qua chi tiết?” ở độ sâu nào?

- Researcher muốn khám phá nhẹ: giải thích hành vi theo điều kiện đầu vào, kiểm tra yếu tố đóng góp bằng can thiệp, hay giải thích cơ chế nội bộ? `[CẦN XÁC NHẬN]` độ sâu mong muốn; chưa có experiment được chốt.
- Chi tiết có thực sự được giữ trong serialization/tokenization và nằm trong input model không?
- Vary shared text trong khi giữ identity evidence có làm sensitivity thay đổi có hệ thống không?
- Shorter input có thực sự giúp model nhạy hơn với chi tiết identity không? Chưa có kết quả, không gọi “bị lấn át” là kết luận.
- Model có đổi score nhưng chưa qua fixed threshold hay gần như không nhạy với thay đổi identity?
- Nếu xét internals, cần evidence/can thiệp lên activations nào? Đọc raw weights hoặc attention map đơn lẻ không tự trả lời nhân quả.
- Nguồn: Trao đổi trực tiếp cuối chat; [nguyên tắc Errudite](../../plans/journals/2026-10-03-errudite-principles-learned.md).

## Q11 — Giới hạn của các phương pháp chuyển từ lĩnh vực khác

- RDA cho phép cả hai nhãn trong binary ER còn dạy được gì? Soft labels có đáng đổi hợp đồng hard labels không?
- UA-MT có uncertainty signal phù hợp khi teacher là LLM riêng thay vì EMA student không?
- SQL/DeMix/PASS/gradient-based reweighting cần supervision hoặc model bổ sung nào, và có giữ được claim hiện tại không?
- IDP-EM cần đối chiếu full text và ablations; lượt tìm trước mới xác nhận publisher summary. Novelty so với ER uncertainty weighting chưa được xác lập.
- Calibration/abstention/conformal giải quyết độ tin cậy theo mục tiêu nào; có đáng đổi full-coverage evaluation không? Chưa chốt.
- Nguồn: [memo uncertainty](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md), [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md).

## Q12 — Bằng chứng còn thiếu và ranh giới diễn giải

- Evidence-check còn ba lỗi literal quote; coverage diagnosis và billing reconciliation là hai việc khác nhau. Có hoàn tất được phần còn lại không? Chưa có quyền gọi thêm trong vòng đã kết thúc.
- Đã reconcile được chi phí ba dispatch lịch sử chưa? Không tự coi known received-response cost là tổng billing cuối.
- Warning tokenizer regex trong lượt test có ảnh hưởng thực không? `[CẦN XÁC NHẬN]`; trao đổi chưa có chẩn đoán nguyên nhân. Một input vượt 4096 đã dẫn đến chạy test với max length 4160; không mô tả mọi setting đối chứng giống hoàn toàn.
- Lightweight ZIP đã đáp ứng lần tải kết quả; mặc định `package-results` có được sửa bỏ weights hay chỉ dùng gói riêng? `[CẦN XÁC NHẬN]`.
- Test WDC đã được xem để hình thành ý tưởng. Cần xác nhận transfer trên dữ liệu chưa dùng để thiết kế; không gọi test cũ là bằng chứng pristine cho phương pháp mới.
- Nguồn: [AGENTS.md](../../AGENTS.md), [hồ sơ prompt](../../documents/261014-student-variant-instruction.md), trao đổi trực tiếp.
