# Ý tưởng và brainstorm

Ngày tổng hợp: 2026-10-19. Chỉ ghi các hướng đã xuất hiện trong trao đổi hoặc memo của trao đổi đó. `chưa thử` nghĩa là chưa ghi nhận thí nghiệm cho ý tưởng; đọc bài hoặc giải thích phương pháp chưa được tính là triển khai. `đang thử` chỉ dùng cho hướng đã có pilot thực tế, không hàm ý có kết quả cải thiện. Không xem danh sách này là kế hoạch được duyệt.

## I01 — Chẩn đoán trước, xử lý theo loại vấn đề

- **Trạng thái:** đang thử.
- **Giả thuyết:** Cặp có nhãn đáng nghi có thể cần các hành động khác nhau; chọn cách xử lý từ bằng chứng có thể tốt hơn prune/relabel đồng loạt.
- **Đã làm:** Evidence-check và quick offline pilot đã có. Hướng nghiên cứu rộng hơn chưa được đánh giá hoàn chỉnh; số thay đổi của pilot còn nhỏ.
- **Chưa chốt:** Policy tự động, các đối chứng, vai trò review và khả năng chuyển dataset.
- **Nguồn:** [D10–D11](decisions.md#d10--chẩn-đoán-bằng-chứng-trước-khi-chọn-xử-lý), [kế hoạch refinement](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/plan.md).

## I02 — Tăng trọng số cặp khó nhưng nhãn đáng tin, lấy cảm hứng JTT

- **Trạng thái:** chưa thử.
- **Giả thuyết:** Tăng ảnh hưởng của các ví dụ hợp lệ mà student chưa học được giúp học khác biệt nhỏ thay vì chỉ dựa vào độ giống chung.
- **Đã bàn:** Lặp mẫu nhiều hơn hoặc nhân loss với weight; JTT nguyên bản lấy lỗi của model ban đầu rồi cố định tập được tăng exposure.
- **Rủi ro đã bàn:** Nhãn teacher sai cũng có thể là mẫu “khó”; tăng weight có thể khuếch đại lỗi. Chưa có factor hay lịch cập nhật được chốt.
- **Nguồn:** [Memo uncertainty liên ngành](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md), [JTT, ICML 2021](https://proceedings.mlr.press/v139/liu21f.html).

## I03 — Similarity/clustering để tìm cặp đáng xem rồi kiểm tra nhãn

- **Trạng thái:** chưa thử.
- **Giả thuyết:** Cặp giống nhau nhưng mang nhãn non-match, hoặc ít giống nhưng mang nhãn match, có thể chứa hard cases hoặc nhãn cần xem lại.
- **Đã bàn:** Tạo danh sách candidate, so sánh/gom nhóm, kiểm tra từng pair; có đề nghị dùng LLM vote ba lần nữa, sửa nhãn rồi upweight.
- **Rủi ro đã bàn:** Similarity không chứng minh identity; ba lần cùng teacher có thể lặp sai; sửa nhầm rồi tăng weight càng hại. Các family lặp lại có thể chiếm ưu thế; hard positives có thể xa theo representation đang dùng.
- **Chưa chốt:** Cách kiểm tra nhãn đủ tin cậy, metric/embedding, cutoff và có relabel bằng LLM hay không. Tách thí nghiệm relabel khỏi upweight là đề nghị của assistant, chưa là giao thức được duyệt.
- **Nguồn:** Trao đổi trực tiếp; [memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md).

## I04 — RHO-LOSS-inspired weighting từ OOF reference

- **Trạng thái:** chưa thử.
- **Giả thuyết:** Cặp mà student hiện tại còn khó nhưng reference không train trên pair đó vẫn ủng hộ nhãn là ứng viên có ích để tăng chú ý.
- **Đã bàn:** Dùng score OOF sẵn có làm reference cố định; so current per-pair loss với reference loss. WDC không nhất thiết train lại reference; dataset sau cần chuẩn bị reference predictions riêng.
- **Giới hạn:** Nguyên bản chọn batch; bounded weighting là adaptation đã được assistant đề xuất. Nếu cả hai model cùng bỏ qua chi tiết, phương pháp có thể không ưu tiên đúng mẫu cần học; nếu cùng học quy tắc sai, nó có thể ủng hộ nhãn sai.
- **Chưa chốt:** Weight/cap/normalization, thời điểm cập nhật, kiểm soát class balance, kiểm tra scale giữa fold và có đưa vào thí nghiệm chính hay không.
- **Nguồn:** [Memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md), [PDF đã tải](../../documents/rho-loss-2022.pdf).

## I05 — Dataset Cartography để phân biệt kiểu khó theo quá trình học

- **Trạng thái:** chưa thử.
- **Giả thuyết:** Theo dõi score theo epoch giúp phân biệt mẫu dễ, mẫu dao động và mẫu luôn khó tốt hơn một score cuối.
- **Giới hạn:** Chưa có per-pair training histories cho phương pháp này. Score dao động hay luôn thấp không tự xác định nhãn đúng hoặc usefulness.
- **Nguồn:** [Memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md), [Dataset Cartography, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.746/).

## I06 — Giảm ảnh hưởng nhãn chưa đáng tin, lấy ý tưởng UA-MT

- **Trạng thái:** chưa thử.
- **Giả thuyết:** Giảm đóng góp training của supervision bất định có thể ít hại hơn bắt student học mọi nhãn mạnh như nhau.
- **Giới hạn:** UA-MT dùng EMA teacher và mask consistency loss trong segmentation; teacher của ta là LLM riêng. Nhân loss với một weight nhỏ là adaptation, không phải triển khai nguyên bản.
- **Nguồn:** [Memo uncertainty](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md).

## I07 — RDA hoặc soft labels để giữ sự không chắc thay vì ép sửa

- **Trạng thái:** chưa thử.
- **Giả thuyết:** Khi chưa biết nhãn nào đúng, cho phép nhiều đáp án hoặc biểu diễn bất đồng có thể tránh ép một nhãn duy nhất.
- **Giới hạn đã bàn:** Với binary ER, RDA cho phép cả hai nhãn có thể làm loss giám sát của pair bằng 0. Soft-label/multi-output đổi hợp đồng hard labels; ba phiếu cùng LLM không phải phân phối ground truth, và chỉ ít pair có split vote.
- **Nguồn:** [Memo uncertainty: RDA, Probabilistic U-Net, soft-label multi-task](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md).

## I08 — Early-Learning Regularization để hạn chế ghi nhớ nhãn sai

- **Trạng thái:** chưa thử.
- **Giả thuyết:** Nếu model học tín hiệu hữu ích trước rồi về sau ghi nhớ nhãn lỗi, bảo toàn phần dự đoán sớm có thể giúp.
- **Giới hạn:** Chưa chứng minh run hiện tại có hiện tượng này; dự đoán sớm cũng có thể sai. OOF cuối không cho thấy diễn biến trong training.
- **Nguồn:** [Memo uncertainty: ELR](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md).

## I09 — Phân tích theo thuộc tính và nhóm tình huống

- **Trạng thái:** chưa thử đối với việc triển khai các phương pháp tham khảo; evidence-check hiện hữu thuộc I01.
- **Giả thuyết:** Định vị thiếu/xung đột ở thuộc tính cụ thể, hoặc nhóm những pair có tình huống chung, giúp đưa ra giả thuyết xử lý rõ hơn.
- **Đã bàn:** SQL node-level uncertainty, DeMix, Domino và nguyên tắc Errudite; nhóm có thể chồng lấp, cần xem cả mẫu đúng hoặc không bị flag.
- **Giới hạn:** Các phương pháp gốc có yêu cầu gold/reference/synthetic error supervision riêng. Chưa chọn đem nguyên detector nào vào pipeline gold-free.
- **Nguồn:** [Nhật ký Domino](../../plans/journals/2026-10-04-domino-suspicious-slice-analysis.md), [nhật ký Errudite](../../plans/journals/2026-10-03-errudite-principles-learned.md), [memo uncertainty](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md).

## I10 — Calibration, selective classification và conformal risk control

- **Trạng thái:** chưa thử.
- **Giả thuyết/khả năng đã bàn:** Calibration làm score phản ánh độ đúng tốt hơn; selective classification cho phép không trả lời một phần mẫu; conformal risk control chọn quy tắc theo mục tiêu rủi ro dưới điều kiện của phương pháp.
- **Giới hạn:** Không tự sửa nhãn teacher hoặc cải thiện ranking. Abstention đổi mục tiêu full-coverage; conformal cần tập calibration và các giả định thích hợp, không tự bảo đảm macro F1.
- **Nguồn:** [Memo uncertainty](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md).

## I11 — Semantic entropy / thêm vote để nhận diện uncertainty

- **Trạng thái:** chưa thử đối với semantic entropy; ba teacher pass đã có là dữ liệu nền khác.
- **Giả thuyết/khả năng đã bàn:** Đo bất đồng giữa các đáp án có cùng nghĩa thay vì giữa các cách viết.
- **Giới hạn:** Nhãn đã chuẩn hóa match/non-match thì semantic clustering thường không thêm nhiều ngoài vote count. Không bắt được sai có hệ thống mà model luôn trả lời giống nhau. Chưa chốt gọi thêm vote.
- **Nguồn:** [Memo uncertainty](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-16-cross-field-uncertainty-methods.md).

## I12 — PASS và meta-reweighting như hướng tham khảo

- **Trạng thái:** chưa thử.
- **Giả thuyết/khả năng đã bàn:** PASS dùng peer agreement để tách supervision đáng tin; Learning to Reweight Examples dùng tác động gradient lên clean validation để học weight.
- **Giới hạn:** PASS cần thay đổi training nhiều model; ba OOF fold không phải ba peer độc lập cho từng pair. Meta-reweighting dùng clean validation trong optimization, khác giao thức gold validation chỉ phục vụ đánh giá/model selection hiện tại.
- **Nguồn:** [Memo hard pairs](../../plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-17-finding-useful-hard-pairs.md).

## I13 — Instruction student phân biệt variant

- **Trạng thái:** bỏ, đối với variant đã thử làm prompt mặc định.
- **Giả thuyết lúc thử:** Làm rõ các xung đột variant sẽ giảm FP trên sản phẩm gần giống.
- **Kết quả và lý do dừng:** Run variant không cải thiện match F1 so với đối chứng refined dùng prompt gốc; researcher yêu cầu rollback. Không mở rộng kết luận thành “mọi prompt cụ thể đều hại”.
- **Nguồn:** [D13](decisions.md#d13--thử-instruction-phân-biệt-variant-rồi-rollback).

## I14 — Tìm hiểu “vì sao student bỏ qua chi tiết” như hướng riêng

- **Trạng thái:** chưa thử.
- **Giả thuyết đã nêu:** Model có thể dựa mạnh vào phần mô tả chung; thông tin xung đột nhỏ có thể bị lấn át. Input, supervision và threshold cũng đã được nêu như các yếu tố cần phân biệt, chưa phải nguyên nhân đã xác nhận.
- **Đã bàn:** Controlled input changes theo CheckList; thay lượng shared text nhưng giữ identity evidence; phân biệt hành vi quan sát được với kiểm tra nguyên nhân và cơ chế nội bộ. Có thể nghiên cứu activations/interventions trong mechanistic interpretability; chưa chốt làm.
- **Giới hạn:** Researcher yêu cầu khám phá nhẹ, riêng. Không cần mặc định đọc raw weights; attention map hay prediction thay đổi một lần không đủ chứng minh cơ chế.
- **Nguồn:** Trao đổi trực tiếp cuối chat; [CheckList, ACL 2020](https://aclanthology.org/2020.acl-main.442/).

## I15 — Prune cặp gần như không có thông tin định danh

- **Trạng thái:** chưa thử đối với quy tắc mới nảy sinh khi xem test errors.
- **Giả thuyết:** Bản ghi không đủ bằng chứng để quyết định identity có thể không cung cấp supervision hữu ích.
- **Đã bàn:** Researcher nêu `49982462#45065236` vì B thiếu thông tin và đồng ý xem xét trường hợp này. Không suy ra đã sửa nhãn hoặc xóa pair.
- **Giới hạn:** Một vài trường trống hoặc bản ghi ngắn không tự đủ lý do prune. Quy tắc dùng cho training và việc có áp dụng thực tế `[CẦN XÁC NHẬN]`; không xóa test errors để cải thiện metric.
- **Nguồn:** Trao đổi trực tiếp; đối chiếu nguyên tắc thiếu thông tin trong [nhật ký xử lý evidence](../../plans/journals/2026-10-06-priority-ordered-evidence-handling.md).

## I16 — Record-order swap: training augmentation và order-flip signal

- **Trạng thái:** hoãn (researcher quyết định 2026-10-22). Probe inference đã chạy; swap training và order-flip detector chưa thử.
- **Giả thuyết:** Student không bất biến theo thứ tự A/B; lỗi flip nhiều hơn pair đúng ở cùng mức confidence. Random 50% swap khi train (kiểu Ditto `entry_swap`) có thể giảm lỗi order-unstable; flip giữa hai order có thể là thêm một uncertainty detector.
- **Đã làm:** Probe trên validation (`analysis/order_swap_probe.py`, commit `4f5b786`); kết quả ở [nhật ký swap](../../plans/journals/2026-10-20-record-order-swap-findings.md). Order fusion +0.7–1.0 pt, chưa kiểm bằng paired test.
- **Lý do hoãn:** Tăng scope (thêm detector, thêm run, thêm logic fusion) trước khi treatment chính có kết quả. Teacher đã rất ổn định (2.487/2.500 pair 3/3), nên phần thêm của swap cho detection còn chưa rõ.
- **Nếu reviewer hỏi:** Bản rẻ là swap chỉ ở student lúc inference, không gọi API trả phí.
- **Nguồn:** Trao đổi trực tiếp 2026-10-22; Phase 3–5 của [plan 261021](../../plans/261021-2235-wdc-order-swap-and-hard-pair-upweighting/plan.md).
