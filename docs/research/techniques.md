# Kỹ thuật và khái niệm đã bàn

Ngày tổng hợp: 2026-10-19. Đây là ghi chú học và cách liên hệ với dự án; nhắc đến một phương pháp không có nghĩa đã triển khai. Trạng thái ý tưởng xem [ideas.md](ideas.md), các lựa chọn đã chốt xem [decisions.md](decisions.md).

## Bài toán và bằng chứng

| Khái niệm | Giải thích ngắn | Dùng trong dự án / giới hạn |
|---|---|---|
| Entity Resolution / Entity Matching | Quyết định hai records có cùng thực thể theo định nghĩa task hay không. | Phân biệt cùng family với cùng specific product/variant. Màu, bundle, phụ kiện hoặc capacity có thể quan trọng tùy identity policy của dataset; cảm nhận “giống nhau” ngoài đời chưa đủ xác định nhãn benchmark. |
| LLM hard-label supervision | LLM cung cấp một nhãn rời rạc cho từng training pair. | So compact model học từ gold với model học từ nhãn LLM; không đưa rationale vào student trong thiết kế hiện tại. |
| Direct LLM baseline | Cho LLM giải quyết trực tiếp các pair cần đánh giá, thay vì train student. | So chất lượng, timing/cost và break-even với compact model theo phạm vi được chốt. |
| Cross-encoder / generative reranker | Xử lý hai records cùng nhau; Qwen reranker tạo logits cho no/yes rồi dùng như binary classifier. | Student có instruction trong config. Đổi instruction khi inference và train lại với instruction mới là hai thí nghiệm khác nhau. [Code](../../models/generative_reranker_student.py). |
| Majority vote / self-consistency | Lấy nhãn đa số từ ba judgment và lưu mức 3/3 hay 2/3. | Đo consistency; không chứng minh correctness. Ba call cùng model có thể cùng sai. |
| OOF / cross-fitting | Mỗi pair được dự đoán bởi model không train trên chính pair đó. | Có một dự đoán held-out cho mỗi pair, không phải ba peer predictions. Tránh direct pair memorization trong reference; chưa tự loại record/family overlap. |
| Teacher–student disagreement | Student và supplied teacher label không đồng ý. | Tín hiệu tìm pair cần xem. Có thể teacher sai, student sai hoặc dữ liệu khó/thiếu; không tự xác định bên nào đúng. |
| Confident Learning / Cleanlab | Dùng nhãn cho sẵn và class probabilities để ước lượng noise, xếp hạng candidate label issues. | Chạy trên OOF; candidate là nghi ngờ, không phải gold error. Class thresholds bên trong CL khác decision threshold dùng khi đánh giá student. [Builder](../../supervision/build_confident_learning.py). |
| Evidence-check / diagnosis | Mô tả tình huống và trích đúng evidence từ records trước khi xử lý. | Các tình huống đã dùng: khác biểu diễn, thông tin một phía, xung đột giữa records, variant/bundle chưa rõ, mâu thuẫn nội bộ, role/scope chưa rõ, dữ liệu nhiễu. Exact quote kiểm tra grounding ở mức chuỗi, chưa xác minh ngữ nghĩa. |
| Prune / relabel / retain | Loại pair khỏi train; đổi nhãn; hoặc giữ nguyên. | Tách riêng artifact/ledger và giữ bản gốc. “Giữ” có thể vì chưa đủ bằng chứng, không có nghĩa nhãn đã xác minh đúng. |
| Hard negative / hard positive | Non-match khó vì rất giống; match khó vì biểu diễn khác nhau. | Khi chỉ có nhãn teacher, đây là “khó theo supervision hiện có”, chưa xác minh lớp thật. Ví dụ code khác một ký tự hay red/blue cần kiểm tra vai trò/ngữ cảnh. |
| Similarity / embedding / clustering | Biểu diễn dữ liệu thành vector, đo gần/xa hoặc gom nhóm. | Dùng tìm ứng viên và mẫu lặp; khoảng cách phụ thuộc representation, không phải khoảng cách identity thật. Cùng sản phẩm khác ngôn ngữ vẫn có thể xa theo metric yếu; không khẳng định khoảng cách thực nếu chưa đo. |

Nguồn trao đổi nền: [kế hoạch refinement](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/plan.md), [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md).

## Đọc kết quả đúng phạm vi

| Khái niệm | Giải thích ngắn | Dùng trong dự án / giới hạn |
|---|---|---|
| FP / FN | Khi match là lớp dương: FP là dataset non-match nhưng student match; FN là dataset match nhưng student non-match. | Báo cáo pair cần đặt nhãn dataset cạnh student prediction và full input; không tự đồng nhất nhãn dataset với chân lý đã được audit. |
| Match F1 / macro F1 | Match F1 tổng hợp precision–recall của lớp match; macro F1 trung bình F1 hai lớp. | Match F1 là metric chính. Macro F1 đang là objective chọn threshold; sự khác nhau này đã được đặt câu hỏi, chưa chốt đổi. |
| Decision threshold tuning | Chọn score cutoff trên validation theo objective rồi dùng cố định trên test. | Quy tắc hiện hữu tối ưu macro F1, rồi match F1, accuracy, gần 0.5 nếu hòa. Có thể overfit validation; không bảo đảm generalization. [Code](../../utils/classification_threshold.py). |
| Score và calibration | Score cao thể hiện model nghiêng mạnh về một nhãn; calibrated score còn cần tương ứng với tỷ lệ đúng thực tế. | Match score trong báo cáo là score student, không phải teacher confidence hay bằng chứng đúng. Threshold thấp không tự chứng minh bug hoặc correctness. |
| Temperature scaling | Điều chỉnh độ sắc của logits trước khi đổi thành xác suất. | Đã bàn như cách calibration; positive temperature trong binary classification giữ nguyên thứ tự score, nên không tự cải thiện attainable precision–recall ranking. Chưa áp dụng. |
| Matched comparison / ablation | Giữ các yếu tố khác tương đương và thay yếu tố đang khảo sát. | Không thay cả nhãn và weighting rồi quy cải thiện cho weighting. Single-pass checkpoint không tự làm baseline cho refined/three-pass. Một run chưa xác định nguyên nhân hay tính ổn định. |
| Provenance / artifact contract | Lưu dataset, labels, prompt, model, runtime và hash để biết kết quả thuộc điều kiện nào. | Giữ config/checkpoint lịch sử; bản variant phải dùng instruction của nó khi tái lập. Không gộp val/test hoặc label sources khác nhau. |
| Break-even | Mức số lượng inference mà chi phí chuẩn bị/train student được bù nhờ inference rẻ hơn direct LLM. | Phải tính cả labeling và các bước OOF/refinement phát sinh. Giá thuê GPU giả định khác khoản API đã đo. |

Nguồn đối chiếu: [hồ sơ prompt](../../documents/261014-student-variant-instruction.md), [nhật ký vertical slice](../journals/260831-1053-wdc-qwen-first-vertical-slice.md), [memo uncertainty](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md).

## Kỹ thuật training đã tìm hiểu, chưa mặc định chọn dùng

### Upweighting và JTT

Nhân loss hoặc tăng số lần thấy selected examples để chúng ảnh hưởng mạnh hơn đến cập nhật model. JTT (ICML 2021) xác định lỗi của model ban đầu rồi train model sau với tập tăng exposure cố định. Trong dự án, ý tưởng là học các khác biệt identity nhỏ; phải đề phòng tăng ảnh hưởng nhãn sai. Dùng OOF errors thay early training errors là adaptation.

### RHO-LOSS

RHO-LOSS (ICML 2022) ưu tiên theo khoảng cách giữa current loss và reference loss. Reference được train trên các ví dụ khác; độ khó còn lại của reference là một ước lượng, không phải phép đo chắc chắn noise không thể học.

Trong trao đổi, ta đã bàn tái dùng `assigned_label_probability` từ [OOF CSV](../../outputs/uncertainty_refinement/wdc-qwen-oof/oof_predictions.csv):

```text
reference_loss = -ln(probability của supplied label)
priority = current_per_pair_loss - reference_loss
```

Với label match thì dùng match probability; với non-match thì dùng non-match probability. Reference cố định có thể đọc từ file, không cần chạy lại model reference. Current loss vẫn phải lấy trong training mới. Chỉ OOF loss không đủ tạo dynamic priority.

Nguyên bản chọn batch; thêm bonus weight có giới hạn là đề xuất adaptation chưa chốt. Hai model cùng bỏ qua chi tiết hoặc cùng học quy tắc sai vẫn là giới hạn. OOF hiện hữu dùng majority labels, không tự đổi thành reference có provenance tương đương với quick-refined labels.

Nguồn: [PDF đã tải](../../documents/rho-loss-2022.pdf), [memo RHO/hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md).

### Các hướng liên ngành còn lại

| Kỹ thuật | Ý chính đã bàn | Liên hệ với dự án / giới hạn |
|---|---|---|
| Dataset Cartography | Đo trung bình và dao động score theo epoch. | Quan sát kiểu học của từng pair; cần history mới, không suy ra từ OOF score cuối. |
| ELR | Dùng dự đoán trung bình trong quá trình học để hạn chế ghi nhớ noisy labels về sau. | Chỉ hợp lý nếu có bằng chứng memorization; dự đoán sớm cũng có thể sai. |
| UA-MT | EMA teacher đo uncertainty để mask consistency supervision không đáng tin ở giai đoạn thích hợp. | Gợi ý giảm ảnh hưởng mẫu bất định; không phải prune vĩnh viễn hoặc nhân toàn bộ loss theo đúng công thức của paper. |
| RDA | Cho phép một tập nhãn khả dĩ thay vì ép một nhãn. | Với hai lớp, cho phép cả hai dưới loss của bài có thể bằng bỏ tín hiệu nhãn của pair; khác soft label 0.5/0.5. |
| Probabilistic outputs / soft-label multi-task | Học nhiều đáp án khả dĩ hoặc phân bố bất đồng annotation. | Khác hợp đồng một hard label; phiếu của cùng LLM chưa phải phân bố người gán nhãn độc lập. |
| Semantic entropy | Gom các câu trả lời cùng nghĩa rồi đo bất đồng giữa nhóm. | Nhãn binary chuẩn hóa đã có hai nhóm; không tự nhận ra sai lặp lại nhất quán. |
| PASS | Hai peer chọn supervision cho model thứ ba qua agreement. | OOF ba fold hiện có không phải committee ba model không thấy từng pair; cần thay đổi training nếu áp dụng. |
| Learning to Reweight Examples | Học weight từ tác động lên loss của clean validation. | Làm gold validation tham gia optimization; không phù hợp để đưa nguyên trạng vào giao thức hiện tại. |
| Selective classification / abstention | Chỉ trả lời những mẫu hệ thống chấp nhận. | Phải báo cả coverage và chất lượng phần đã trả lời; khác F1 full-coverage hiện tại. |
| Conformal risk control | Chọn quy tắc kiểm soát expected risk với calibration data và điều kiện thích hợp. | Chưa có giao thức calibration cho dự án; không bảo đảm từng pair đúng hoặc tự tối ưu macro F1. |

Nguồn và venue đã tra nằm trong [memo uncertainty liên ngành](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md) và [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md). Các paper như Block-SCL/Sudowoodo cung cấp tiền lệ mining hard pairs; không tự xác minh nhãn LLM. IDP-EM mới được kiểm tra publisher summary trong lượt nghiên cứu đó, chưa xác nhận chi tiết công thức.

## Khung điều tra lỗi và câu hỏi “vì sao”

| Khung/khái niệm | Giải thích ngắn | Cách dùng đã bàn |
|---|---|---|
| Errudite | Định nghĩa pattern rõ, xem đủ nhóm liên quan, kiểm tra giải thích thay vì chỉ kể ví dụ. | Xem cả pair không bị flag hoặc model làm đúng; cần mẫu số, không lấy đặc điểm phổ biến trong nhóm lỗi làm nguyên nhân. |
| Slice discovery / Domino | Tìm nhóm có đặc điểm chung mà model hoạt động kém. | Với teacher labels chỉ có thể gọi nhóm bất đồng/đáng nghi; nguyên bản dùng true labels, không tự chuyển thành detector gold-free. |
| CheckList / behavioral testing | Kiểm tra năng lực bằng thay đổi input có kiểm soát. | Thay thuộc tính identity và so với thay đổi vô hại như viết hoa; đo score lẫn decision. Chủ yếu chỉ ra hành vi xảy ra khi nào. |
| Controlled intervention | Thay yếu tố nghi ngờ và giữ evidence quan trọng để xem nó có góp phần gây hành vi không. | Ví dụ thay lượng shared text nhưng giữ chi tiết identity. Kết quả có thể hỗ trợ một yếu tố đóng góp, chưa chỉ ra nguyên nhân duy nhất hoặc cơ chế bên trong. |
| Mechanistic interpretability | Nghiên cứu biểu diễn và các phép tính nội bộ, thường qua activations và can thiệp. | Hướng khám phá riêng cho câu hỏi model biểu diễn/bỏ qua color ở đâu. Chưa triển khai; đọc raw weights hoặc attention map đơn lẻ chưa đủ giải thích nhân quả. |
| Quan sát / giả thuyết / kết luận | “Có xung đột capacity” là quan sát; “model bỏ qua capacity” là giả thuyết. | Phải kiểm tra trên mẫu khác hoặc qua can thiệp; không lấy chính ví dụ tạo giả thuyết làm toàn bộ bằng chứng xác nhận. |

Nguồn: [nhật ký Errudite](../../plans/journals/2026-10-03-errudite-principles-learned.md), [nhật ký mental models](../../plans/journals/2026-10-04-analysis-mental-models-and-hypothesis-validation.md), [CheckList](https://aclanthology.org/2020.acl-main.442/), trao đổi trực tiếp cuối chat về câu hỏi “vì sao”.
