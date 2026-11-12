# Các quyết định đã bàn và đã chốt

Ngày tổng hợp: 2026-10-19; cập nhật D18–D21 ngày 2026-10-22, D22 ngày 2026-10-24, D23 ngày 2026-10-28, D24 ngày 2026-11-04, D25 ngày 2026-11-06, D26 ngày 2026-11-12. Bản này ghi lại trao đổi hiện có và các nhật ký trong repo; không thay thế giao thức thí nghiệm hoặc cấp quyền chạy thêm. Ngày dưới đây là ngày quyết định nếu có căn cứ; ngày không chắc được đánh dấu `[CẦN XÁC NHẬN]`.

Các đề nghị của assistant chưa được researcher chốt được đặt trong [ideas.md](ideas.md). Vấn đề chưa giải quyết nằm trong [open-questions.md](open-questions.md). Không tự điền lý do hay phương án bị loại nếu lịch sử không ghi nhận.

## D01 — Giữ Qwen sau screening lịch sử

- **Ngày:** 2026-07-24.
- **Quyết định:** Cho Qwen3-Reranker-0.6B qua vòng screening, tạm dừng tìm kiến trúc và đi tiếp với Qwen trong giai đoạn khi đó.
- **Lý do:** Các điều kiện screening dự đoán được cả hai lớp, chất lượng validation đủ để đi tiếp; tránh tình trạng chỉ dự đoán một lớp từng gặp ở chẩn đoán ModernBERT.
- **Phương án đã loại hoặc hoãn:** Tiếp tục tìm kiến trúc ngay lúc đó được hoãn. Lặp nhiều seed cũng được hoãn để đi tiếp; không được diễn giải lợi thế active selection lúc ấy là kết quả ổn định.
- **Giới hạn:** Đây là quyết định của thiết kế low-label cũ, không chốt đủ ba model cho thiết kế hiện tại.
- **Nguồn:** [Nhật ký screening](../journals/260724-1537-qwen-screening-phase6-decision.md).

## D02 — Chuyển sang so sánh full-label trên ba dataset và ba model

- **Ngày:** 2026-08-20.
- **Quyết định:** Thiết kế chính là 3 dataset × 3 compact cross-encoder × 2 nguồn nhãn train: gold và LLM hard labels; một run khai báo trước cho mỗi ô, kèm direct-LLM baseline theo dataset. Match F1 là chỉ số chính; tốc độ, chi phí và break-even là bằng chứng bổ sung.
- **Lý do:** Nghiên cứu khả năng dùng nhãn LLM thay nhãn train benchmark và giảm chi phí triển khai so với gọi LLM trực tiếp. Lý do đầy đủ của việc bỏ thiết kế cũ: `[CẦN XÁC NHẬN]`; không suy ra chỉ từ thao tác migration.
- **Phương án đã loại:** Low-label budgets và active-selection workflow cũ được cho nghỉ khỏi luồng chính. Rationale distillation, cascade, thêm dataset/model và chiều repeated-run không nằm trong phạm vi đã ghi.
- **Giới hạn:** Tên chính xác của toàn bộ dataset/model và hợp đồng chung chưa chốt xong. Hướng refinement là phần mở rộng đã được bàn riêng.
- **Nguồn:** [Kế hoạch chính](../../plans/260820-1507-full-label-er-migration/plan.md), [nhật ký migration](../journals/260820-1440-full-label-migration-cleanup.md).

## D03 — Dùng chung OpenRouter và cố định upstream

- **Ngày:** 2026-08-22.
- **Quyết định:** Screening dùng client OpenRouter chung, cố định OpenAI upstream, tắt fallback, nhận nhãn có cấu trúc thay vì rationale.
- **Lý do:** Tránh lệch transport/cách tính chi phí giữa các workflow; giữ điều kiện model/provider so sánh được; kiểm tra đầu ra rõ ràng.
- **Phương án đã loại:** Client OpenAI riêng cho screening được bỏ để tránh trùng logic. Không tự fallback sang provider khác vì sẽ đổi điều kiện chạy.
- **Nguồn:** [Nhật ký migration labeler](../journals/260822-1347-openrouter-labeller-screening-migration.md).

## D04 — Chọn Sol-high cho WDC và làm một lát cắt đầu-cuối trước

- **Ngày:** Đã chọn trước hoặc vào 2026-08-24; ngày chốt chính xác `[CẦN XÁC NHẬN]`.
- **Quyết định:** Chọn Sol-high sau screening ba setting; hoàn thành full-label WDC trước, rồi so sánh Qwen train bằng gold và LLM-hard.
- **Lý do:** Sol-high được chọn vì cân bằng chi phí/chất lượng. Lát cắt WDC giúp có bằng chứng đầu-cuối trước khi mở rộng hợp đồng toàn bộ ma trận.
- **Phương án đã loại:** `sol_max` và `sol_pro_max` không được chọn làm setting production WDC. Không ghi rằng chúng luôn kém chính xác hơn: lý do lựa chọn là cân bằng, không phải một kết luận phổ quát.
- **Nguồn:** [AGENTS.md](../../AGENTS.md), [nhật ký full labeling](../journals/260824-1652-wdc-sol-high-full-labeling.md).

## D05 — Tách nhãn benchmark khỏi việc tạo và sửa nhãn LLM

- **Ngày:** Được ghi trong thiết kế 2026-08-20 và các trao đổi refinement sau đó.
- **Quyết định:** Giữ official splits; không đưa gold vào request labeler hoặc dùng gold để quyết định candidate/refinement. Gold train thuộc arm gold hoặc đánh giá sau khi đóng băng đầu ra; gold validation/test dùng cho đánh giá theo giao thức.
- **Lý do:** Tránh đưa giám sát benchmark vào arm được tuyên bố dùng nhãn LLM. Validation có thể phục vụ checkpoint/decision-threshold selection đã quy định; điều đó khác với dùng gold validation để học trọng số từng mẫu.
- **Phương án đã loại:** Sửa trực tiếp các bất đồng LLM–gold bằng gold rồi vẫn gọi đó là nhãn LLM độc lập. Không lấy test làm nơi chọn phương pháp/ngưỡng.
- **Giới hạn:** Test WDC của thí nghiệm prompt về sau đã được researcher cho chạy và đã được xem. Không áp nguyên trạng câu “test chưa mở” của các nhật ký cũ lên mọi thí nghiệm về sau.
- **Nguồn:** [Kế hoạch chính](../../plans/260820-1507-full-label-er-migration/plan.md), [kế hoạch refinement](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/plan.md), [thí nghiệm prompt](../../documents/261014-student-variant-instruction.md).

## D06 — Chọn structured DBLP-ACM làm dataset thứ hai

- **Ngày:** Hợp đồng nguồn được chốt 2026-09-02.
- **Quyết định:** Dùng snapshot structured DBLP-ACM từ DeepMatcher, đối chiếu ZIP chính thức với bản tải độc lập; giữ split, identity và serialization theo hợp đồng nguồn.
- **Lý do:** Cần căn cứ từ dữ liệu tải và audit tại chỗ, không kế thừa các quan sát snapshot chưa kiểm chứng. Tuổi của benchmark được ghi là hạn chế.
- **Phương án đã loại:** Các quan sát snapshot kế thừa bị researcher bác ngày 2026-09-01. Không sửa các luồng WDC đã hash-bound chỉ để tích hợp DBLP vì sẽ làm lệch hợp đồng tái lập.
- **Giới hạn:** Chuẩn bị dữ liệu/fake run không đồng nghĩa đã có production labels hay kết quả model trên DBLP.
- **Nguồn:** [Nhật ký DBLP-ACM](../journals/260831-dblp-acm-integration-plan.md).
- **Trạng thái:** Đã được thay bằng D24 (2026-11-04); DBLP-ACM không còn là dataset 2.

## D07 — Tạo ba teacher pass và dùng majority labels làm đầu vào OOF

- **Ngày:** Hướng được ghi ngày 2026-09-28; ngày duyệt từng lần chạy xem hồ sơ gốc.
- **Quyết định:** Giữ một teacher, tạo ba lượt judgment riêng cho training pairs; lưu majority label và độ đồng thuận riêng, không sửa input gốc.
- **Lý do:** Bổ sung tín hiệu consistency để khảo sát độ bất định của nhãn. Pass mới phải là judgment mới, không được tái dùng nhãn cũ rồi tính thành phiếu mới.
- **Phương án đã loại:** Không coi việc gọi lại cache là một pass độc lập. Không coi kết quả single-pass lịch sử tự động là arm majority/refined tương đương.
- **Giới hạn:** Ba phiếu cùng teacher không bảo đảm độc lập hoặc đúng; đồng thuận vẫn có thể cùng bỏ qua một chi tiết.
- **Nguồn:** [Kế hoạch refinement](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/plan.md).

## D08 — OOF theo pair, dùng Qwen làm detector chính

- **Ngày:** 2026-09-30; hoàn tất chạy/merge được ghi 2026-10-02.
- **Quyết định:** Ba fold, đơn vị chia là `pair_id`; 20% phần không held-out dùng làm inner validation. Qwen làm detector chính; các model còn lại vẫn là đối tượng downstream evaluation.
- **Lý do:** Cần xác suất cho mỗi training pair từ model không train trên chính pair đó để phục vụ phát hiện nhãn đáng nghi; tránh mở rộng thành toàn bộ ma trận detector theo student.
- **Phương án đã loại:** Không dùng dự đoán sau khi train trên chính pair làm bằng chứng OOF. Không mặc định có ba dự đoán độc lập cho mỗi pair chỉ vì chạy ba fold.
- **Giới hạn:** Pair-level OOF không tự bảo đảm record hoặc product family không xuất hiện ở các fold khác.
- **Nguồn:** [Kế hoạch refinement](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/plan.md).

## D09 — Dùng Cleanlab để xếp hạng đáng nghi, chưa tự sửa nhãn

- **Ngày:** 2026-10-02.
- **Quyết định:** Dùng Cleanlab 2.9.0 trên majority labels và xác suất OOF; công bố candidate/ranking tách khỏi dữ liệu gốc.
- **Lý do:** Có một tín hiệu nghi ngờ nhãn không dùng gold để tiếp tục khảo sát.
- **Phương án đã loại:** Không biến mọi candidate thành lỗi đã xác minh hoặc tự động prune/relabel chỉ vì bị đánh dấu. Chưa chốt cách fusion với teacher consistency.
- **Giới hạn:** Số candidate và số lỗi ước lượng là hai đại lượng khác nhau; Cleanlab và teacher–student disagreement cùng dùng OOF nên không phải hai xác nhận độc lập.
- **Nguồn:** [Nhật ký CL](../../plans/journals/2026-10-02-wdc-cleanlab-oof-diagnostics.md).

## D10 — Chẩn đoán bằng chứng trước khi chọn xử lý

- **Ngày:** Hướng được ghi 2026-10-02; evidence-check được cho thực hiện 2026-10-05.
- **Quyết định:** Khảo sát loại vấn đề trước khi chọn giữ, prune hoặc relabel; evidence-check nhận records và định nghĩa tình huống, không nhận nhãn cũ, score hoặc cờ chọn mẫu. Giữ raw responses và trích dẫn nguyên văn.
- **Lý do:** Bất đồng có thể đến từ nhãn sai, thiếu dữ kiện hoặc mẫu khó hợp lệ; không có một cách sửa mặc nhiên phù hợp mọi trường hợp.
- **Phương án đã loại:** Đưa thẳng mọi suspicious pair sang một hành động như thể mọi pair đều sai. Không coi quote hợp lệ về mặt chuỗi là bảo đảm diễn giải ngữ nghĩa đúng.
- **Giới hạn:** Đây là chẩn đoán do model sinh, không phải nguyên nhân hay nhãn đúng đã được kiểm chứng.
- **Nguồn:** [Nhật ký hướng refinement](../../plans/journals/2026-10-02-diagnosis-guided-suspicious-label-refinement.md), [nhật ký evidence-check](../../plans/journals/2026-10-05-wdc-teacher-evidence-case-classification.md).

## D11 — Quick-refinement offline, bảo thủ với trường hợp khó

- **Ngày:** 2026-10-06.
- **Quyết định:** Cho làm pilot offline bằng majority labels, diagnoses sẵn có và ledger assistant review; giữ nguyên trường hợp khó/chưa phân tích/diagnosis không hợp lệ. Pilot prune case 5 trước và chỉ đổi nhãn ở quyết định xung đột rõ đã ghi.
- **Lý do:** Muốn có một bước thử nhanh từ bằng chứng hiện hữu mà không gọi thêm LLM; tránh ép quyết định ở trường hợp chưa đủ rõ.
- **Phương án đã loại:** Không dùng bộ từ khóa ngữ nghĩa tự động để thay review. Không gọi thêm evidence-check trong bước pilot này. Prune mọi case 5 là lựa chọn riêng của pilot, không được nâng thành quy luật đã chứng minh.
- **Giới hạn:** Pilot có 7 prune và 1 relabel; không đủ để kết luận đã cải thiện student hoặc đã sửa đúng toàn bộ nhãn. Quyết định theo từng pair có assistant review, không phải pipeline ngữ nghĩa hoàn toàn tự động.
- **Nguồn:** [Nhật ký và follow-up pilot](../../plans/journals/2026-10-06-priority-ordered-evidence-handling.md).

## D12 — Báo cáo lỗi cần đặt nhãn dataset cạnh dự đoán student và đầy đủ input

- **Ngày:** Trong lượt xem báo cáo khoảng 2026-10-13; ngày trao đổi chính xác `[CẦN XÁC NHẬN]`.
- **Quyết định:** Researcher yêu cầu khoảng 30 pair, mỗi pair hiển thị nhãn từ dataset, dự đoán/score student và full model input. “Counterpart” được làm rõ là dữ liệu trong dataset, không tự hiểu thành judgment LLM khác.
- **Lý do:** Muốn phân biệt vấn đề của dữ liệu/định nghĩa nhãn với lỗi student, đồng thời đọc được bằng chứng của cả hai records.
- **Phương án đã loại:** Chỉ đưa tổng FP/FN hoặc chỉ tiêu đề rút gọn không đáp ứng mục tiêu kiểm tra này. Ví dụ viết “dataset match, student non-match” phải gọi là FN khi match là lớp dương; không tiếp tục nhãn FP nhầm trong ví dụ trao đổi.
- **Giới hạn:** Pair `49982462#45065236` làm nảy sinh đề nghị prune vì B thiếu thông tin; chưa ghi nhận một thay đổi training dataset từ đề nghị này.
- **Nguồn:** Trao đổi trực tiếp trong chat; báo cáo đã nhắc ở `outputs/analysis/` là artifact phân tích, không phải tập nhãn mới.

## D13 — Thử instruction phân biệt variant, rồi rollback

- **Ngày:** Thử nghiệm được lưu trong ghi chú 2026-10-14; rollback theo yêu cầu researcher ngày 2026-10-15.
- **Quyết định:** Đã thử instruction student nhấn mạnh model/color/size/capacity/pack/bundle; sau đó khôi phục instruction ban đầu cho công việc tiếp theo và giữ config/checkpoint variant làm bằng chứng lịch sử.
- **Lý do:** FP ở sản phẩm gần giống gợi ý thử làm rõ task cho student. Kết quả test match F1 của variant là 0.7465986395, thấp hơn 0.7581227437 ở đối chứng refined dùng prompt gốc. Researcher muốn bỏ prompt variant và để instruction mở hơn cho model khái quát hóa; đây là động cơ, chưa phải nguyên nhân được chứng minh.
- **Phương án đã loại:** Không tiếp tục dùng variant làm prompt mặc định. Không thay instruction của variant-trained checkpoint rồi gọi đó là tái lập kết quả cũ. Checkpoint single-pass trước đó không phải đối chứng cùng phạm vi của refined/three-pass.
- **Giới hạn:** FP tăng 188 lên 237, FN giảm 80 xuống 61. Một run không chứng minh prompt dài nói chung kém hơn hoặc toàn bộ giả thuyết refinement thất bại. Test đã góp phần hình thành hướng này nên không còn là xác nhận hoàn toàn chưa được xem.
- **Nguồn:** [Hồ sơ prompt và rollback](../../documents/261014-student-variant-instruction.md).

## D14 — Giữ cách chọn decision threshold hiện hữu, chưa chốt đổi quy tắc

- **Ngày:** Trao đổi sau thí nghiệm prompt; ngày chính xác `[CẦN XÁC NHẬN]`.
- **Quyết định:** Quy tắc đang dùng là chọn threshold trên validation để tối đa macro F1; hòa thì xét match F1, accuracy, rồi gần 0.5. Threshold đã lưu được dùng nguyên trên test. Chưa có quyết định thay bằng mức sàn 0.3/0.5 hoặc tối ưu theo test.
- **Lý do đã bàn:** Tách lựa chọn operating point khỏi test; score chưa chắc đã calibrated nên cutoff có thể rất thấp. Ai/ngày nào chốt macro F1 thay vì match F1 ban đầu: `[CẦN XÁC NHẬN]`.
- **Phương án đã loại hoặc chưa chốt:** Không ghi rằng researcher đã đồng ý đổi objective để cứu kết quả. Không đồng nhất threshold này với threshold của Confident Learning.
- **Giới hạn:** Researcher đặt câu hỏi về overfit validation/cherry-picking và khả năng generalization; đây vẫn là vấn đề cần kiểm chứng, không được coi là đã bác bỏ.
- **Nguồn:** Trao đổi trực tiếp; [code chọn threshold](../../utils/classification_threshold.py).

## D15 — Gói kết quả nhẹ để tải, không kèm weights

- **Ngày:** Trong lượt lấy kết quả prompt; ngày chính xác `[CẦN XÁC NHẬN]`.
- **Quyết định:** Researcher yêu cầu package phục vụ tải kết quả chỉ có predictions và metrics, tránh đưa weights vào ZIP.
- **Lý do:** Gói trước đang kèm weights; mục tiêu lần tải này là phân tích kết quả.
- **Phương án đã loại:** Gói lớn kèm weights cho nhu cầu đọc metrics/predictions này. Không suy ra rằng mọi runner đã được sửa mặc định packaging; trạng thái thay đổi mặc định `[CẦN XÁC NHẬN]`.
- **Nguồn:** Trao đổi trực tiếp; [hồ sơ runner lịch sử](../../documents/261014-student-variant-instruction.md) ghi rõ packaging cũ có weights.

## D16 — Ưu tiên tìm hard-but-useful pairs; chưa chọn thuật toán cuối

- **Ngày:** Trao đổi được tổng hợp 2026-10-16–17.
- **Quyết định:** Researcher muốn ưu tiên một cách hệ thống để nhận diện cặp khó nhưng hữu ích và tăng trọng số, có thể dùng lại ở dataset sau. RHO-LOSS được yêu cầu giải thích và tải để đọc.
- **Lý do:** Có quan sát student bỏ qua khác biệt nhỏ như màu hoặc ký tự mã; researcher không muốn thêm một bước judgment LLM rồi mặc nhiên tin nó.
- **Phương án đã loại hoặc chưa chốt:** Không coi code-distance hoặc similarity là bộ phán quyết ngữ nghĩa. Chưa có quyết định chọn RHO-LOSS, JTT, công thức weight hay chạy train mới.
- **Giới hạn:** Kế hoạch refinement cũ chỉ ghi prune/relabel, chưa gồm reweight. Việc đưa weighting vào thí nghiệm chính `[CẦN XÁC NHẬN]`.
- **Nguồn:** Trao đổi trực tiếp; [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md).

## D17 — Khám phá câu hỏi “vì sao” như một hướng nhẹ, riêng

- **Ngày:** Lượt trao đổi trước bản tổng hợp này; ngày chính xác `[CẦN XÁC NHẬN]`.
- **Quyết định:** Researcher nói rõ muốn khám phá nhẹ câu hỏi vì sao model hành xử như vậy, không chỉ tìm một cách sửa lỗi tiếp theo.
- **Lý do:** Muốn hiểu cơ chế hoặc nguyên nhân của việc bỏ qua chi tiết.
- **Phương án đã loại hoặc chưa chốt:** Không mặc định biến hướng này thành thí nghiệm chính của luận văn. Chưa chốt đào sâu weights/activations hay triển khai mechanistic interpretability.
- **Giới hạn:** Assistant đã phân biệt kiểm tra hành vi, kiểm tra yếu tố gây ảnh hưởng bằng can thiệp, và phân tích cơ chế nội bộ; chưa thực hiện phép kiểm chứng nào cho hướng mới này.
- **Nguồn:** Trao đổi trực tiếp trong chat.

## D18 — Câu hỏi chính chuyển sang tìm và xử lý uncertain training pairs

- **Ngày:** Researcher nêu trong Project trước 2026-10-21; ngày chốt chính xác `[CẦN XÁC NHẬN]`. Ghi vào bản này 2026-10-22.
- **Quyết định:** Câu hỏi chính của luận văn là làm sao tìm các training pair bất định trong nhãn LLM và xử lý chúng thế nào. Ba tín hiệu phát hiện đã có; trọng tâm hiện tại là cách xử lý.
- **Lý do:** Mục tiêu cốt lõi là để LLM gán nhãn dữ liệu thay con người. Gold của train chỉ dùng ở bước cuối để kiểm chứng, không dùng để thiết kế hay chọn phương pháp.
- **Giới hạn:** Thiết kế full-label 3 dataset × 3 model (D02) còn là khung chính hay chỉ là nền: `[CẦN XÁC NHẬN]`. Validation/test dùng nhãn người vì cần đúng, và rẻ hơn gán nhãn train; WDC validation (2.500) bằng kích thước train nên lập luận "rẻ hơn" yếu ở WDC (xem `REVISIT.md`).
- **Nguồn:** Trao đổi trực tiếp; [AGENTS.md](../../AGENTS.md).

## D19 — Phát hiện: giữ teacher–student disagreement và teacher 2/3

- **Ngày:** 2026-10-22.
- **Quyết định:** Tập uncertain pairs là hợp của 149 OOF teacher–student disagreement và 13 pair teacher 2/3: 158 pair, cả 158 đã có evidence-check diagnosis (155 hợp lệ, 3 invalid). Confident Learning chỉ là phân tích phụ.
- **Lý do:** Researcher: trong khung distillation, bất đồng giữa teacher và student hợp lý hơn làm tín hiệu chính. Kiểm tra cùng ngày: 120/120 CL candidate nằm trong 149 disagreement, nên CL không phải tín hiệu độc lập.
- **Phương án đã loại:** Thay disagreement bằng CL (được đồng ý sớm trong cùng trao đổi, sau đó researcher đổi ý). Thêm record-order swap làm detector thứ tư được hoãn ([ideas.md I16](ideas.md)).
- **Giới hạn:** Disagreement chỉ đánh dấu pair bất định, không chứng minh nhãn sai.
- **Nguồn:** Trao đổi trực tiếp; `outputs/uncertainty_refinement/wdc-qwen-cl/cl_candidates.csv`, `outputs/uncertainty_refinement/wdc-qwen-oof/oof_predictions.csv`.

## D20 — Xử lý: guideline do người viết, LLM judge áp dụng

- **Ngày:** 2026-10-22.
- **Quyết định:** Người (researcher) định nghĩa category và guideline quyết định; LLM judge đọc hai record và guideline, không thấy nhãn cũ, và trả về: quyết định được hay không, match/non-match, trích dẫn nguyên văn. Một quy tắc cố định chuyển kết quả thành hành động: không quyết định được → remove; quyết định được và trùng teacher → upweight (hard-but-useful); quyết định được và khác teacher → flip. Judge chỉ chạy trên 158 uncertain pairs.
- **Lý do:** Category là đa nhãn (đa số pair có 3–4 category; `one_sided_information` ở 138/155) nên không tự quyết định được hành động; ví dụ xung đột mã model khác xung đột giá. Người viết guideline, không viết nhãn, giống annotation guideline. Áp quy tắc lên output đã lưu cho kết quả tất định dù LLM ngẫu nhiên. Chỉ chạy chọn lọc vì judge tốn khoảng USD 0,019/pair, khoảng 30× GPT-5.2 labeling của Steiner & Bizer (arXiv:2606.28823) trên WDC; chạy toàn bộ 2.500 pair (~USD 48/pass) sẽ vượt chi phí họ gán nhãn cả năm benchmark.
- **Phương án đã loại:** Ánh xạ chỉ từ category sang hành động (assistant đề xuất). 65 quyết định assistant review của quick pilot (D11) làm phương pháp, vì không tái lập được. Chạy judge trên mọi training pair.
- **Giới hạn:** Guideline chưa viết; chưa chọn judge model (cùng teacher hay khác) và số pass; paid run cần researcher cho phép; cần giảng viên duyệt không `[CẦN XÁC NHẬN]`. Duplication k = 2 và random control lấy từ plan 261021, sẽ chốt lại trong contract mới. Judge cùng model với teacher có thể bị coi là vote thứ tư (xem `reviewers-question-mock/q02`).
- **Nguồn:** Trao đổi trực tiếp; [nhật ký tier](../../plans/journals/2026-10-06-priority-ordered-evidence-handling.md); [evidence-case prompt](../../supervision/evidence_case_prompt.md); `reviewers-question-mock/q01-judge-on-all-pairs.md`.

## D21 — Plan 261021 vào backlog; swap hoãn

- **Ngày:** 2026-10-22.
- **Quyết định:** `plans/261021-2235-wdc-order-swap-and-hard-pair-upweighting/` chuyển sang backlog (status `backlog`, mục trong `REVISIT.md`). Swap training và order-flip detector hoãn ([ideas.md I16](ideas.md)).
- **Lý do:** Ngoài phạm vi câu hỏi hiện tại. Cách chọn hard pair theo RHO-style/teacher-trust trên toàn bộ train (K = 250) được thay bằng xử lý theo guideline trên 158 uncertain pairs (D20).
- **Giữ lại:** Gold boundary, duplication k = 2, random control cùng cỡ và cùng tỉ lệ lớp, paired bootstrap + McNemar, test chạy một lần cho arm đã đóng băng.
- **Nguồn:** Trao đổi trực tiếp; `REVISIT.md`.

## D22 — Version 1 xử lý bằng category rule trên diagnoses đã có; guideline v3 được duyệt

- **Ngày:** 2026-10-23 đến 2026-10-24.
- **Quyết định:** Guideline judge v3 (`supervision/judge_guideline.md`) được researcher duyệt và đóng băng ngày 2026-10-24: không phụ thuộc domain, không có domain profile, step 4 theo option B (thông tin một phía không chặn quyết định), không có ngưỡng giá cố định, đầu ra gồm verdict/step/reasoning/evidence. Version 1 không gọi judge mới: áp các bước của guideline lên category của 155 diagnoses hợp lệ đã có, rồi áp action rule (remove / upweight / flip). Arms A (nhãn teacher), B (bỏ cả 158 pair), C (category rule). Arm D bỏ ở version 1.
- **Lý do:** Researcher tự trả chi phí nên muốn làm cẩn thận và rẻ; diagnoses đã có cho biết pair thuộc category nào nên có thể áp quy tắc trực tiếp; kết quả tất định và trích dẫn đã được kiểm nguyên văn.
- **Phương án đã loại hoặc hoãn:** Paid judge run cho version 1. Judge panel gồm nhiều model khác nhau (ví dụ GPT, Claude, Kimi) để làm version 2, chỉ trên các pair ở step 3, sau khi có kết quả version 1. Domain profile và hướng dẫn viết profile ([I17](ideas.md)). Quy tắc chặn match theo tỉ lệ giá 2×.
- **Giới hạn:** Step 3 loại mọi pair mơ hồ (86, trong đó 51 teacher match); step 6 nghĩa là "không thấy vấn đề", không phải "đã xác nhận cùng sản phẩm"; diagnoses do cùng họ model với teacher sinh ra (`reviewers-question-mock/q02`). Giảng viên không cần duyệt trước (researcher, 2026-10-23).
- **Nguồn:** Trao đổi trực tiếp; [plan 261023](../../plans/261023-2015-wdc-guideline-judge-treatment/plan.md), [contract](../../plans/261023-2015-wdc-guideline-judge-treatment/contract.md).

## D23 — Dataset 3: MaDI-Bench Companies, chỉ subtask entity matching

- **Ngày:** 2026-10-28.
- **Quyết định:** Dataset 3 là MaDI-Bench Companies (base, commit `062058d`), chỉ bước entity matching với split chính thức, gold schema mapping của benchmark, giá trị raw chỉ làm sạch nhẹ, quy tắc threshold macro-F1 hiện hành (D14), một model cho cả hai source pair và báo cáo F1 theo từng source pair cùng trung bình. Giữ DBLP-ACM vì là dataset duy nhất không đến từ nhóm Mannheim (phần này đã được thay bằng D24).
- **Lý do:** Nhãn matching được công bố sau knowledge cutoff của teacher (GPT-5.6 Sol, 16/02/2026), nên teacher không thể đã nhớ gold; thêm domain công ty mà bài Steiner & Bizer chưa kiểm tra.
- **Giới hạn:** Benchmark là preprint (arXiv:2606.30371 v1), repo không có license; FullContact cấm phân phối lại, nên dữ liệu không được đưa lên git hay công bố, chỉ báo cáo kết quả tổng hợp. Train/val có lỗi nhãn đã thấy; quy tắc parent/subsidiary không nhất quán; test Forbes–DBpedia nhỏ (140 cặp). Bản ghi nguồn cũ (2012–2014) nên LLM có thể biết các công ty nổi tiếng.
- **Nguồn:** [Hồ sơ quyết định](dataset3-madi-companies.md); [plan 261028](../../plans/261028-2230-madi-companies-dataset3/plan.md).

## D24 — Dataset 2: thay DBLP-ACM bằng Geo-ER Pittsburgh (OSM–Foursquare)

- **Ngày:** 2026-11-04.
- **Quyết định:** Dataset 2 là Geo-ER Pittsburgh, source pair OSM–Foursquare (Balsebre và cộng sự, WWW 2022, NTU Singapore), dùng split chính thức `data/train_valid_test/osm_fsq/pit/{train,valid,test}.txt` (2.500 / 999 / 1.502 cặp, khoảng 29% match ở mọi split). Tiền xử lý theo cùng quy tắc Companies: tách `COL <field> VAL <value>` thành name, latitude, longitude, address, postalCode; giá trị raw, chuỗi rỗng và `NULL` thành `<missing>`; A = OSM, B = Foursquare. Threshold theo quy tắc macro-F1 hiện hành (D14) trên toàn bộ validation; match F1 là metric chính. Thay đổi này sửa hợp đồng Dataset 2 DBLP-ACM đã duyệt (D06).
- **Lý do:** Nhãn DBLP-ACM đã cũ (công khai từ khoảng 2010) và 7.417 cặp train quá tốn để gán nhãn và huấn luyện trong ngân sách. Dataset thay thế phải đến từ nhóm ngoài Mannheim (WDC Products và MaDI-Bench Companies đều từ Mannheim) và có split chính thức khoảng 2.000–2.500 cặp train; Pittsburgh là thành phố duy nhất gần cỡ này. Domain địa điểm là mới, và bằng chứng quyết định là vị trí, nên kiểm tra được guideline độc lập domain.
- **Phương án đã loại hoặc dự phòng:** DBLP-ACM (D06). OSM–Yelp cùng thành phố (có 1 cặp trùng và 1 cặp nằm ở cả val và test). Machamp Books/Movies (Megagon Labs, BSD-3) là dự phòng nếu Geo-ER thất bại, ví dụ student không học được từ tọa độ.
- **Giới hạn:** Tiêu chí teacher cutoff được nới cho dataset này: nhãn công khai từ 2022 nên teacher có thể đã thấy. Bản ghi mỏng (5 trường, khoảng một nửa address trống), mọi cặp đã qua bộ lọc tên tương tự, nên khoảng cách phải suy từ tọa độ raw; student nhỏ có thể không làm được. Nhãn từ bốn annotator, không báo cáo độ đồng thuận. Repo không có license; dữ liệu Foursquare có điều khoản hạn chế, nên xử lý như FullContact: không commit hay công bố bản ghi, chỉ báo cáo kết quả tổng hợp, thêm một câu về điều khoản dữ liệu trong luận văn.
- **Khoảng cách (chốt 2026-11-04):** Thêm trường `distance_m` vào mỗi cặp ngay từ đầu, không chờ pilot. Đây là biến đổi tất định từ tọa độ hai bản ghi, không dùng nhãn; tọa độ raw vẫn được giữ. Áp dụng như nhau cho mọi arm và split, và teacher LLM cũng thấy cùng input (teacher và student đọc đúng một chuỗi input, như WDC và Companies). Đây là ngoại lệ có chủ đích so với quy tắc chỉ giá trị raw, phải nêu trong luận văn như một bước tiền xử lý. Lý do: student nhỏ khó tự suy khoảng cách từ tọa độ dạng chữ, và mục tiêu là kết quả tốt nhất có thể (researcher: "we will need to do all the tricks to get better result anyway").
- **Đã làm (2026-11-04):** Adapter `data/prepare_geo_er.py` (pair id theo số dòng trong file split chính thức; khóa test), runner gán nhãn `supervision/run_full_labeling_geo_er.py` và runner arm `scripts/run_geo_er_arms.sh`. Số cặp, số match và phân bố `distance_m` trên train khớp với số liệu researcher đã đo.
- **Còn mở:** Cập nhật lịch: giai đoạn DBLP-ACM 3 tuần thành giai đoạn Geo-ER.
- **Nguồn:** [Hồ sơ quyết định](dataset2-geo-er-pittsburgh.md); repo https://github.com/PasqualeTurin/Geo-ER; DOI 10.1145/3485447.3512026.

## D25 — Một prompt evidence check chung, trung tính về domain, cho mọi dataset

- **Ngày:** 2026-11-06.
- **Quyết định:** Dùng một prompt evidence check chung cho WDC, Companies và Geo-ER: giữ quy tắc và bảy nhóm tình huống của prompt WDC, bỏ cách diễn đạt và ví dụ riêng cho sản phẩm, không có đoạn riêng cho từng dataset. Chỉ danh sách trường được trích dẫn đi theo schema của từng dataset. Prompt `supervision/evidence_case_prompt_generic.md` được researcher ký duyệt ngày 2026-11-06 (SHA-256 `48b2c4615336f09dc6240a04a8d71182b32d81c8306dd10048f8adb8dce8ec2c`): bảy nhóm tình huống mỗi nhóm một câu, không đoạn riêng cho dataset, giải thích không bắt buộc tiếng Việt (trường `note`, `summary`, `uncertainties`); `{FIELDS}` lấy từ schema. Runner chung `supervision/run_evidence_check_v2.py`, cấu hình teacher chung `configs/labelers/evidence_check_sol_high_flex.json` (flex tier, giá trần 2/10).
- **Lý do:** Prompt WDC chứa ví dụ và đoạn riêng cho sản phẩm (SKU, giá, màu, pack size), vô nghĩa với công ty và địa điểm. Vì WDC đằng nào cũng phải chạy lại để mọi dataset dùng cùng một prompt, chọn phương án đơn giản nhất: một prompt cho tất cả.
- **Hệ quả:** WDC chạy lại evidence check với prompt mới (khoảng USD 3,3, flex tier), dựng lại bundle C, huấn luyện lại arm C và test một lần như một arm đóng băng mới (C v2), kèm amendment cho contract plan 261023. Arm A, B, G, OOF và 158 cặp phát hiện của WDC không đổi. Kết quả arm C cũ được giữ làm bản pilot.
- **Phương án đã loại:** Prompt WDC nguyên văn cho mọi dataset; một lõi chung kèm khối domain theo template (đã cân nhắc rồi bỏ cùng ngày).
- **Căn cứ:** Steiner & Bizer (arXiv:2606.28823, mục 2.3) chọn mô tả nhiệm vụ general-complex của Peeters et al. (2023) vì cách diễn đạt độc lập domain, dùng nguyên văn cho cả sản phẩm và dữ liệu thư mục.
- **Nguồn:** Trao đổi 2026-11-06; [contract plan 261106](../../plans/261106-1330-companies-treatment-arms/contract.md); amendment A5 của [contract plan 261023](../../plans/261023-2015-wdc-guideline-judge-treatment/contract.md).

## D26 — Các run còn lại chạy trên Google Colab A100, fp16

- **Ngày:** 2026-11-12.
- **Quyết định:** Vì nền tảng thuê V100 cũ bảo trì không rõ ngày mở lại, mọi run huấn luyện còn lại chạy trên Google Colab với GPU A100 (pay-as-you-go), precision fp16, cùng recipe. Tên GPU thực tế được ghi trong `runtime.json` của từng run và `EXPECTED_GPU` đặt theo đúng card đó.
- **Lý do:** Không có V100 sẵn trong một tuần; Colab A100 rẻ (khoảng 5–8 unit/giờ, khoảng USD 0,5–0,75/giờ), không cần thuê máy, dữ liệu chạy trong datacenter của Google. Researcher: đổi card gây khác biệt ngẫu nhiên nhưng không lớn, như mọi người làm AI đều biết.
- **Giới hạn và cách giảm:** Khác biệt do card có thể cùng cỡ với hiệu ứng đang đo (khoảng 1 điểm F1). Vì vậy hai arm được so sánh với nhau phải cùng chạy trên A100: chạy lại arm tham chiếu (A, nếu cần cả G) một lần trên A100 trước khi so sánh, và các seed mới của mọi arm đều chạy trên A100. Kết quả seed 42 trên V100 (WDC A, B, C, G; Companies A, G) giữ nguyên, báo cáo là kết quả V100.
- **Dữ liệu:** Colab chạy trên máy datacenter của Google nên dùng được cho cả Companies và Geo-ER: chỉ để dữ liệu trong thư mục Drive riêng tư, không chia sẻ, không đưa lên repo công khai, xoá sau khi xong.
- **Nguồn:** Trao đổi 2026-11-12; amendment A6 của [contract plan 261023](../../plans/261023-2015-wdc-guideline-judge-treatment/contract.md).

