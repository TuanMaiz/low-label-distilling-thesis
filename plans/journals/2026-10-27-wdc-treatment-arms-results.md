# Kết quả ba arm A/B/C (WDC–Qwen) và đánh giá "gần nhãn người với chi phí thấp"

Ngày: 2026-10-27

Trạng thái: arm A, B (26/10) và C (27/10) đã train, đánh giá validation và test một lần theo plan `plans/261023-2015-wdc-guideline-judge-treatment/`. Chưa có paired test, chưa có gold-label arm cùng recipe, mỗi arm một seed. Mục 3 tóm tắt một research report bên ngoài; các đề xuất trong đó chưa phải quyết định.

## 1. Thiết lập chạy

- GPU: Tesla V100-SXM2-32GB (thuê), fp16 (contract A3). Python của template ở `/opt/conda/bin/python`, không nằm trong PATH; torch 2.2.1, CUDA 12.1 (giống OOF tháng 9). Runbook: `docs/runbooks/rented-v100-treatment-arms.md`.
- Ba arm cùng commit `a60d486`, seed 42, LoRA, effective batch 16, lr 2e-4, tối đa 10 epoch, early stopping patience 3, không cắt input.
- C chạy trên máy thuê mới (27/10). Trước setup đã pin `transformers 4.57.6, peft 0.21.2, accelerate 1.15.0, tokenizers 0.22.2, safetensors 0.8.0, huggingface_hub 0.36.2, sentencepiece 0.2.2`. `pip freeze` của C và A không có khác biệt nào.
- Kiểm tra từng gói tải về (`outputs/newABC/arm-{A,B,C}.zip`): checksum khớp; 12/12 file adapter khớp checkpoint manifest (merged model cố ý không đóng gói); F1 tính lại từ predictions khớp metrics; threshold khớp training summary; test contract đúng arm.
- Validation match F1 lúc train (epoch tốt nhất) chênh khoảng 0,001 so với lần evaluate riêng (A: 0,8261 so với 0,8250). Báo cáo dùng số của lần evaluate.

## 2. Kết quả (một seed mỗi arm)

Validation: 2.500 cặp (500 match). Test: 4.500 cặp (500 match), 100% sản phẩm chưa thấy.

| Split | Arm | TP | FP | FN | Match P | Match R | Match F1 | Macro F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | A (nhãn teacher, không xử lý) | 448 | 138 | 52 | 0,7645 | 0,8960 | **0,8250** | 0,8883 | 0,9240 |
| Validation | B (bỏ cả 158 cặp) | 445 | 154 | 55 | 0,7429 | 0,8900 | 0,8098 | 0,8781 | 0,9164 |
| Validation | C (category rule) | 436 | 152 | 64 | 0,7415 | 0,8720 | 0,8015 | 0,8731 | 0,9136 |
| Test | A | 450 | 226 | 50 | 0,6657 | 0,9000 | **0,7653** | 0,8650 | 0,9387 |
| Test | B | 451 | 250 | 49 | 0,6434 | 0,9020 | 0,7510 | 0,8564 | 0,9336 |
| Test | C | 442 | 240 | 58 | 0,6481 | 0,8840 | 0,7479 | 0,8549 | 0,9338 |

| Arm | Train rows | Epoch chạy | Epoch tốt nhất | Decision threshold | GPU-h |
|---|---:|---:|---:|---:|---:|
| A | 2.500 | 10 | 7 | 1,81×10⁻⁴ | 1,92 |
| B | 2.342 | 9 | 6 | 1,33×10⁻⁴ | 1,63 |
| C | 2.459 | 9 | 6 | 2,42×10⁻⁶ | 1,69 |

Quan sát (chưa kiểm định):

- A cao nhất ở gần như mọi chỉ số trên cả hai split. B và C chủ yếu mất precision (nhiều false match hơn); C mất thêm recall (validation 0,872 so với 0,896).
- Giả thuyết của assistant: step 3 loại 86 cặp mơ hồ, trong đó 51 cặp teacher match, làm mất tín hiệu ở các corner case. Chưa kiểm chứng.
- Threshold của C rất thấp (2,4×10⁻⁶), cho thấy phân bố score bị dịch khi nhãn đổi.
- Chênh 1,4–1,7 F1 giữa A và B/C nằm trong khoảng nhiễu giữa các seed được báo cáo cho cấu hình tương tự, nên hiện chỉ nói được "chưa có bằng chứng xử lý giúp ích", không nói được "xử lý làm tệ hơn".
- Test WDC đã được dùng cho A, B, C. Mọi arm mới thiết kế sau khi xem kết quả này phải chọn trên validation; DBLP-ACM là tập xác nhận sạch.

## 3. Research report: "gần nhãn người với chi phí thấp" có đủ không?

Researcher nêu: luận văn không cần vượt nhãn người, chỉ cần gần với chi phí thấp hơn nhiều. Research report (27/10, nguồn bên ngoài) kết luận: hướng này có triển vọng nhưng hiện chưa đủ để khẳng định. Các điểm chính:

- **Tài liệu:** với entity matching, student học từ nhãn LLM thường cách student học từ nhãn benchmark khoảng ±2 F1. Steiner & Bizer (arXiv:2606.28823 v1): WDC Ditto 71,94 (nhãn benchmark) so với 72,17 (nhãn máy tốt nhất), trung bình ± SD qua ba seed trở lên. Report ghi có bản v2 (tháng 10/2026, 8 benchmark, "under submission to PVLDB"), số liệu khác v1; khi trích dẫn cần ghi rõ phiên bản. Ở các bài NLP khác khoảng cách thường 1–4 F1; Pangakis & Wolken 2024 báo median gap 0,039 F1, student nhãn GPT có recall cao hơn nhưng precision thấp hơn, giống mẫu P/R của arm A.
- **Thiếu gì:** (a) gold-label arm cùng recipe (V100, fp16, cùng thiết lập); số 0,821 của 31/08 dùng bf16 trên RTX 3090 nên không so trực tiếp được; (b) nhiều seed; (c) margin non-inferiority khai báo trước, kiểm bằng paired bootstrap trên ΔF1.
- **Non-inferiority:** không chứng minh được "bằng nhau"; chỉ chứng minh được "không tệ hơn quá δ". Report đề xuất δ = 2 F1 (khớp mức "dưới hai điểm F1" của Steiner & Bizer và cỡ nhiễu seed); kết luận được nếu cận dưới khoảng tin cậy 95% của ΔF1 = F1(LLM) − F1(gold) lớn hơn −2. McNemar chỉ là kiểm tra phụ.
- **Chi phí:** với giả định 30 giây/cặp (Steiner & Bizer), 2.500 cặp mất khoảng 20,8 giờ người; report ước khoảng 167–333 USD ở 8–15 USD/giờ cộng phí nền tảng, so với khoảng 8,1 USD cho ba lượt LLM, tức khoảng 20–40×; kịch bản bi quan (10 giây/cặp, 8 USD/giờ) vẫn khoảng 7×. Nên trình bày dạng bảng độ nhạy, nêu rõ phần nào được tính và vẫn cần test set do người gán.
- **Reviewer có thể hỏi:** gold baseline cùng recipe; chênh lệch có phải nhiễu seed; có chọn arm theo test không; ai đúng ở 79 cặp LLM–gold bất đồng (gold WDC có khoảng 1,91% lỗi theo audit của Steiner & Bizer); vì sao precision chỉ 0,666; khả năng teacher đã thấy WDC lúc pretrain; chi phí gồm những gì.
- **Về B và C:** report xem là kết quả null có thông tin ("không có bằng chứng rằng hậu xử lý dựa trên uncertainty giúp ích"), phù hợp Lu & Smith 2025 và Pangakis & Wolken 2024.
- Report tự ghi giới hạn: chưa đọc hết full text Steiner & Bizer (một số header bảng, số cặp audit và kiểm định dùng chưa xác minh); khoảng tin cậy ±2–3 F1 cho một model là ước lượng thô; "≥ 5 seed" là nhận định, không phải chuẩn có nguồn.

Nguồn chính: [Steiner & Bizer v1](https://arxiv.org/html/2606.28823v1), [abstract](https://arxiv.org/abs/2606.28823), [repo](https://github.com/wbsg-uni-mannheim/Automatic-data-labeling), [WDC Products](https://arxiv.org/abs/2301.09521), [Wang et al. 2021](https://arxiv.org/pdf/2108.13487), [Pangakis & Wolken 2024](https://arxiv.org/pdf/2406.17633), [Lu & Smith 2025](https://arxiv.org/html/2504.15432v1), [Lakens 2017](https://journals.sagepub.com/doi/10.1177/1948550617697177), [Reimers & Gurevych 2017](https://aclanthology.org/D17-1035.pdf), [Dror et al. 2018](https://aclweb.org/anthology/P18-1128.pdf), [Card et al. 2020](https://arxiv.org/pdf/2009.03037).

## 4. Trao đổi và quyết định trong ngày

- Làm rõ: arm A **không** phải gold; cả A, B, C đều học từ nhãn LLM. Gold là nhãn người trong `data/cache/wdc_products/full_label_targets/gold.jsonl`.
- Researcher: **3 seed là đủ** (khớp "three or more" của Steiner & Bizer). Kế hoạch tối thiểu do assistant đề xuất, chưa chốt: gold arm cùng recipe × 3 seed, thêm 2 seed cho A; khoảng 5 run, khoảng 9–10 giờ V100. Gold arm dùng train gold làm mốc tham chiếu, ngoài contract hiện tại nên cần amendment trước khi chạy.
- Margin δ = 2 F1 và cách kiểm non-inferiority: assistant giải thích, researcher chưa chốt.
- Nếu C không tốt hơn (đã thấy ở single seed), hướng đã bàn: (1) phân tích train-gold các thay đổi của C; (2) tách C thành remove-only / flip-only / upweight-only; (3) version 2: judge panel cho 86 cặp step 3; (4) xác nhận trên DBLP-ACM. Chưa chốt.

## 5. Việc tiếp theo

1. `analysis/paired_comparison.py`: paired bootstrap ΔF1 và McNemar (C vs A, C vs B, sau này A vs gold).
2. Phân tích train-gold cho các hành động của C (contract cho phép ở bước cuối).
3. Amendment cho gold arm và thêm seed; chốt δ trước khi chạy.
