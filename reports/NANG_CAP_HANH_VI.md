# Kết quả nâng cấp đặc trưng hành vi người và xe

## Đã triển khai

Phiên bản mới dự đoán 17 điểm tư thế bằng YOLOv8n Pose, kết hợp các khớp vai, khuỷu, cổ tay, hông, đầu gối, mắt cá và điểm đầu nhìn thấy được với quan hệ người–xe. Có **67 giá trị mỗi thời điểm**, gộp mean/std/last thành **201 giá trị cho mỗi cửa sổ người dài tối đa 4 giây**. Không dùng màu áo, mặt hoặc danh tính để quy kết hành vi.

| Nhóm | Đại lượng đã trích xuất | Giới hạn |
|---|---|---|
| Tư thế | Tọa độ chuẩn hóa theo thân người, góc khuỷu, độ nghiêng thân, vị trí đầu tương đối | Góc 2D phụ thuộc camera; không xác định ý định |
| Bàn tay | Khoảng cách cổ tay tới vùng xe, tốc độ cổ tay tương đối thân, thời gian tay ở gần vùng trên xe | Vùng trên bounding box xe chỉ là vùng thay thế, chưa định vị ổ khóa/tay lái; không có keypoint ngón tay |
| Quan hệ người–xe | Khoảng cách, thời gian cùng cặp, số tiếp cận/chuyển xe quan sát được | Che khuất/đổi track ID có thể làm sai bộ đếm; đứng cạnh xe chưa đồng nghĩa tiếp xúc |
| Chuyển động | Tốc độ người, đổi hướng, tốc độ xe và đồng chuyển động | RANSAC affine bù chuyển động camera; không tính bằng chứng chuyển động xe khi bù không tin cậy |
| Chất lượng | Mask từng khớp theo confidence, mask chuyển động, số quan sát | Confidence ≥ 0,4 chỉ là ước lượng của pose model, không bảo đảm khớp thực sự nhìn thấy; khớp dưới ngưỡng giữ trạng thái thiếu |

Phần chuyển động cổ tay yêu cầu các mốc thân người tin cậy ở cả hai thời điểm để tránh coi thay đổi điểm neo là thao tác tay. Đổi xe, mất quan sát lâu hoặc cắt cảnh làm reset các đặc trưng tích lũy liên quan. Tốc độ người/xe được chuẩn hóa theo chiều cao người/đường chéo vùng xe trên ảnh và giây, không phải m/s thực địa. Thống kê theo thời gian có thể mô tả động tác lặp, nhưng chưa phải bộ nhận dạng có nhãn cho từng động tác phá khóa, mở cốp hoặc dắt xe.

## Huấn luyện và so sánh

Mỗi video tạo một bag các cửa sổ theo dõi người. Head MIL tuyến tính học bằng nhãn **cả video**, max-pooling các cửa sổ để tính loss. Không gán mọi người hoặc mọi frame trong video Stealing thành kẻ trộm. Scaler chỉ fit train, mỗi video có trọng số cân bằng khi fit scaler. Chọn epoch bằng loss validation, chọn ngưỡng bằng F1 validation, rồi chọn cấu hình bằng F1 và AUC validation.

| Mô hình | F1 validation | F1 test cũ | AUC test cũ |
|---|---:|---:|---:|
| scene baseline | 0.769 | 0.500 | 0.650 |
| pose_only | 0.667 | 0.545 | 0.575 |
| interaction_only | 0.615 | 0.625 | 0.750 |
| pose_and_interaction | 0.667 | 0.571 | 0.650 |

Cấu hình chọn trong họ MIL: **pose_only**. Baseline vẫn là mặc định vì F1 validation của mô hình mới chưa vượt baseline. Test cũ đã được xem trước đó; các số test ở bảng chỉ là chẩn đoán sau phát triển, chưa phải ước lượng độc lập về độ chính xác thực tế. Không chọn interaction_only chỉ vì nó có AUC test cao nhất; cách đó sẽ dùng test để chọn mô hình.

Đã trích xuất trên 63 video. Chỉ 31/40 video train có cửa sổ người đủ số quan sát; 9 video không có mẫu sử dụng được. Đây là nút thắt detector/che khuất/lấy mẫu. Các video test thiếu mẫu vẫn được giữ trong đánh giá với điểm 0, không bị loại để làm đẹp kết quả. Tệp coverage.csv ghi chi tiết từng video. Bộ dữ liệu vẫn là Stealing/Normal nói chung, chưa phải nhãn trộm xe máy trong hầm.

## Dùng trên giao diện

Tải lại trang, chọn video thật, vào Theo dõi trực tiếp trên video và bật **Trích xuất tư thế và tương tác người–xe**. Khung xương và bảng bằng chứng hiển thị độ nhìn rõ khớp, tay gần vùng xe và chuyển động tay. Tốc độ mục tiêu hiển thị là 4 FPS; đặc trưng hành vi cập nhật 2 lần/giây. Bảng Kết quả mô hình có so sánh mới.

Checkbox **Dùng điểm MIL hành vi cho cảnh báo thử nghiệm** mặc định tắt. Khi bật, điểm MIL chỉ được dùng khi đủ quan sát, đủ khớp nhìn rõ, gần xe và vượt ngưỡng hai lần cập nhật liên tiếp. Ngưỡng được chọn cho nhãn video, chưa được kiểm định cho sự kiện từng người; không dùng chế độ này để khẳng định độ chính xác vận hành. Các giá trị MIL trong khung chỉ là điểm học giám sát yếu, chưa hiệu chuẩn thành xác suất phạm tội. Phiên bản này chưa kiểm chứng tracking ID, pose hoặc thời gian cảnh báo bằng nhãn đối tượng/sự kiện.

## Tái lập

```bash
python scripts/setup_detector.py
python scripts/extract_behavior.py
python scripts/train_behavior.py
python scripts/summarize_behavior.py
python scripts/annotate_video.py data/external_test/tuoitre_ham_chung_cu_20230731.mp4 --behavior --out runtime/annotated/tuoitre_pose.mp4
python -m pytest -q
```

Cache nằm ở data/behavior_features; model/ablation/coverage tại artifacts/behavior_v2. Model baseline artifacts/model.joblib được giữ nguyên. Source và hash pose weights được lưu riêng.

## Dữ liệu tiếp theo quyết định khả năng dùng thực tế

Hai video YouTube bạn cung cấp được giữ làm đối chứng ngoài, không đưa vào train hoặc chọn ngưỡng. Chúng mô tả bãi xe quán và bãi giữ xe ngoài trời, chưa phải hầm chung cư. Nhãn bình thường là nhãn giả định theo ngữ cảnh cung cấp, chưa có xác minh từng người có quyền lấy xe. Kết quả riêng nằm ở external_evaluation.json.

| Video đối chứng | Baseline cấp video | MIL lấy mẫu | Khung cần xem xét khi duyệt toàn bộ |
|---|---:|---:|---:|
| [Camera Ghi Hình Bãi Giữ Xe Ban Đêm Quán Cháo Cối MT68](https://www.youtube.com/watch?v=a1bcB0_RMiQ) | 0.263 | 0.569 | 97 |
| [Lắp Đặt Camera Quan Sát Bãi Giữ Xe  -  KAMERA](https://www.youtube.com/watch?v=5BSCCBLD-_M) | 0.722 | 0.587 | 105 |

Nếu tạm coi hai video là bình thường, MIL tạo cảnh báo ở cả hai; baseline cấp video vượt ngưỡng ở một video. Đây là lỗi báo nhầm cần xử lý, không phải bằng chứng độ chính xác đã tăng. Số khung vượt quy tắc không phải số sự kiện độc lập hoặc precision/recall. Chưa có nhãn quyền lấy xe và nhãn sự kiện độc lập, vì vậy không công bố false-alarm/hour như một kết quả đã kiểm định. Các video vẫn nằm ngoài train và chọn ngưỡng.


Cần dữ liệu hầm cùng camera cho cả hành vi hợp lệ và nghi vấn: tìm chìa, khóa kẹt, mở cốp, sửa xe, nhân viên dắt xe, đứng chờ, đi ngang và các tình huống dàn dựng có phép. Gán nhãn khoảng thời gian, người–xe và quyền di chuyển được xác minh; chia theo camera/buổi quay trước khi cắt clip. Video đã dùng phân tích lỗi cần được coi là dữ liệu phát triển; phải có test mới khóa riêng. Đo precision/recall sự kiện, báo giả/camera/giờ và độ trễ thay vì chỉ accuracy video.

Tăng số đặc trưng chưa đủ để phân biệt chủ xe và người không được phép nếu động tác hình ảnh giống nhau. Khi cần quyết định quyền lấy xe, phải có xác minh của bảo vệ hoặc tín hiệu cấp quyền như thẻ xe. Chưa có bằng chứng mô hình mới chính xác hơn khi vận hành thực tế.

## Nguồn kỹ thuật

- [Ultralytics Pose và keypoints](https://docs.ultralytics.com/tasks/pose/)
- [ST-HOI về kết hợp tư thế, quỹ đạo và tương tác theo thời gian](https://arxiv.org/abs/2105.11731)
- [Các giới hạn của nhận dạng bất thường bằng tư thế](https://arxiv.org/abs/2303.05463)

Mã hiện tại là baseline MIL nhỏ của đồ án, không tái hiện hoặc nhận kết quả của các bài nghiên cứu trên.
