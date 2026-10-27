# Phát hiện người nhỏ trong video cổng hầm

## Nguyên nhân kiểm tra được

Video upload `592a131ee67e3a47.mp4` có kích thước **424×240 pixel**. Tại 2,5 giây, YOLOv8n không phát hiện người ở cả kích thước suy luận 640 và 1280 với ngưỡng 0,15. Checkpoint bổ sung YOLO11 có ứng viên người confidence khoảng 0,207 ở kích thước 1280, và khoảng 0,24–0,26 khi chia ảnh thành các vùng phóng lớn. Độ tin cậy thấp nên cần xem lại khung phát hiện, chưa thể suy ra người đã được nhận đúng hoặc có hành vi trộm.

Thông báo chờ 4 giây là thời gian tích lũy **điểm cảnh**. Detector chạy từ frame đầu tiên, không cần đợi 4 giây; không có khung ở giai đoạn này có thể là bỏ sót của detector. Khi không có người được phát hiện, video mới ghi `NO PERSON DETECTED | Behavior not assessed` để tránh hiểu nhầm.

## Cách chạy chế độ mới trên web

1. Khởi động lại Streamlit và chọn video.
2. Chọn **Theo dõi trực tiếp trên video**.
3. Tại **Model phát hiện người và xe**, chọn **Model bổ sung: yolo_behavior_model.pt (YOLO11 COCO)**.
4. Tại **Cách chạy detector**, chọn **Người nhỏ / bị che: toàn ảnh 1280 + 4 vùng phóng lớn**.
5. Lần đầu đối chiếu nên giữ ảnh gốc, tắt lọc nhiễu/làm nét để tránh thay đổi nhiều yếu tố cùng lúc. Có thể giữ phân tích tư thế.
6. Tắt **Phát theo thời gian gốc** nếu muốn xử lý nhanh nhất có thể, bấm **Phân tích**. Máy yếu có thể xử lý chậm hơn tốc độ video.

Video cũ không tự có thêm khung sau khi cập nhật code hoặc đổi lựa chọn. Phải bấm **Phân tích** để tạo kết quả mới. Giao diện mặc định vẫn là detector nhanh để không làm chậm mọi video.

## Pipeline mới

- Chạy toàn ảnh với kích thước suy luận 1280.
- Chia ảnh thành 4 vùng chồng lấn, mỗi vùng rộng/cao khoảng 65% ảnh, suy luận ở 640. Model được nhìn đối tượng ở tỷ lệ lớn hơn.
- Quy đổi tọa độ từng vùng về tọa độ ảnh gốc. Không vẽ khung theo tọa độ crop.
- Lọc trùng bằng NMS riêng từng lớp, IoU 0,45; không gộp người với xe máy dù hai khung chồng lên nhau.
- Giữ ứng viên từ confidence 0,15; dùng ByteTrack sau khi gộp. Ngưỡng bắt đầu track là 0,20. Chỉ những track đang được tracker trả về mới nhận ID, ứng viên còn lại có ID rỗng.
- Ứng viên người confidence dưới 0,35 hoặc không có ID **chưa đủ đánh giá hành vi**, hiện xám. Không dùng ngưỡng phát hiện thấp để bỏ qua điều kiện chất lượng của cảnh báo.

Lựa chọn này tăng từ 1 lên tối đa 5 lần suy luận mỗi frame, gồm một lần ảnh lớn, nên chậm hơn. Có thể tăng cả bỏ sót được khắc phục lẫn phát hiện nhầm, đặc biệt nhầm xe/người ngồi trên xe. Phóng ảnh không tái tạo chi tiết không tồn tại trong video 240p và không nhìn xuyên xe/cột che khuất.

Không thay đổi mô hình đã train, nhãn hành vi hoặc tự đánh dấu người là trộm. Metadata trong JSON ghi `detector_metadata.profile`, ngưỡng confidence, tên file và hash trọng số để đối chiếu chính xác cấu hình đã chạy.

## Kết quả chạy lại đúng video

Cùng checkpoint bổ sung, giữ ảnh gốc và bật tư thế, nguồn điểm cảnh. Cả hai lần đều xử lý 120 frame ở 4 FPS, video output mở đọc lại được.

| Chỉ số | Nhanh | Người nhỏ |
|---|---:|---:|
| Frame có ứng viên người | 12/120 | 78/120 |
| Tổng lượt ứng viên người–frame | 12 | 113 |
| Thời gian xử lý trên máy thử | 9,52 giây | 42,29 giây |
| Frame có cảnh báo hành vi | 0 | 0 |
| Tại đúng 2,5 giây | Không có người | Một ứng viên confidence 0,218, chưa có ID |

Các lần chạy có thể chịu tải máy khác nhau; thời gian không phải benchmark hiệu năng cố định. Chế độ mới đồng thời đổi kích thước suy luận, chia vùng, ngưỡng phát hiện và ngưỡng tracker. Chưa tách riêng tác động của mỗi yếu tố.

Confidence thấp và thiếu tư thế/ID khiến ứng viên vẫn xám, chưa được suy ra nguy cơ trộm. Số ứng viên tăng có thể bao gồm phát hiện nhầm, không phải recall hoặc accuracy. Kết quả chi tiết: [small_people_verification.json](../artifacts/small_people_verification.json). Video thử trên máy ở `runtime/annotated/small_people_gate.mp4`.

## Chạy bằng Terminal

```bash
python scripts/annotate_video.py runtime/uploads/592a131ee67e3a47.mp4 --detector-weights model/yolo_behavior_model.pt --detector-profile small_people --behavior --backend scene --no-evidence --out runtime/annotated/small_people_gate.mp4
```

Đường dẫn upload chỉ tồn tại trên máy đã tải video đó; máy khác thay bằng đường dẫn video của mình. `--no-evidence` dùng khi kiểm tra kỹ thuật để không tạo cảnh báo trong sổ; bỏ tùy chọn này khi cần thử chụp ảnh tự động.
