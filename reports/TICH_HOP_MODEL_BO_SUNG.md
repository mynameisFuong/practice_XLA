# Tích hợp checkpoint yolo_behavior_model.pt

## Kiểm tra checkpoint

- File: `model/yolo_behavior_model.pt`, khoảng 5,4 MB; nạp được bằng Ultralytics YOLO.
- Tác vụ: `detect`; kiến trúc trong metadata là `yolo11n.yaml`.
- Có 80 tên lớp COCO: lớp 0 `person`, lớp 3 `motorcycle`. Không có lớp hành vi ăn trộm/bẻ khóa.
- Metadata train ghi `data=/usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml`. Thông tin này mô tả checkpoint nhận được, chưa xác minh quá trình huấn luyện bên ngoài.
- Các metrics nằm trong checkpoint là metrics phát hiện đối tượng trên dữ liệu của lần train đó; không đại diện độ chính xác cảnh báo trộm của hệ thống này.

## Tích hợp

Giao diện thêm lựa chọn detector riêng, giữ YOLOv8n làm mặc định để dễ so sánh. Model bổ sung dùng cùng ByteTrack, trích xuất tư thế và tương tác, nguồn điểm cảnh/pose MIL/ResNet MIL, quy tắc ba mức, video có khung và bộ chụp ảnh như pipeline hiện tại. Bộ kiểm tra từ chối checkpoint không có tác vụ và ánh xạ lớp tương thích, tránh dùng nhầm nhãn.

`stream_video` nhận thêm `detector_weights`. CLI có `--detector-weights`. JSON kết quả lưu đường dẫn, SHA-256, tác vụ, tên lớp và tracker trong `detector_metadata`. Bộ nạp Streamlit kiểm tra giao diện mới để tránh lỗi module cũ thiếu tham số. Gói nộp bài gồm cả thư mục `model`.

## Cách test

1. Mở web, chọn video thật và chế độ **Theo dõi trực tiếp trên video**.
2. Chọn **Model bổ sung: yolo_behavior_model.pt (YOLO11 COCO)** trong **Model phát hiện người và xe**.
3. Bật trích xuất tư thế, chọn nguồn điểm muốn thử. Để đối chiếu detector, giữ nguyên nguồn điểm qua hai lần chạy.
4. Có thể tắt **Phát theo thời gian gốc** để xử lý nhanh hơn, bấm **Phân tích**.
5. Quan sát người/xe bị bỏ sót, khung sai và ID đổi; tải video và JSON để xem lại. Nếu bật chụp ảnh, cảnh báo vàng/đỏ được lưu như trước.
6. Chạy lại cùng video với **YOLOv8n hiện tại**. Muốn đo precision/recall hoặc độ chính xác hành vi phải có nhãn đối chứng và khoảng thời gian sự kiện; số khung phát hiện không phải accuracy.

Các lựa chọn chỉ áp dụng sau khi bấm **Phân tích**. Chế độ **Phân tích tổng quan** chưa sử dụng detector bổ sung.

## Kiểm chứng kỹ thuật

Kết quả smoke test hai detector trên cùng đoạn video thật 6 giây, bật tư thế và giữ nguồn điểm `scene`, được lưu tại [custom_detector_verification.json](../artifacts/custom_detector_verification.json). Video và nhật ký chi tiết nằm trong `runtime/custom_detector_test/` trên máy chạy. Đoạn thử không có nhãn đánh giá; chỉ kiểm tra nạp model, theo dõi, chạy pipeline và xuất video đọc được, không chứng minh cải thiện độ chính xác.
