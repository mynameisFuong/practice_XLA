# Thử tích hợp ResNet50 từ repo tham khảo và đánh giá ba mức

## Kiểm tra nguồn

Đã kiểm tra [repo Dilaniu](https://github.com/Dilaniu/Theft-Detection-Project-using-OpenCV) tại commit `c4383d497a3a51dbda385121a6a52d925d8b401f`. Snapshot chỉ có notebook, README và LICENSE, không có checkpoint trộm riêng. Hàm quyết định dùng đúng hai khuôn mặt và khoảng cách tọa độ; ResNet50 chưa được nối vào quyết định. Class COCO 1 là bicycle, motorcycle là 3. Vì vậy không sử dụng nguyên quy tắc này. Phản ví dụ bằng tọa độ tổng hợp được lưu ở `artifacts/upstream_review/audit.json`; không chạy notebook Colab hoặc nhận dạng danh tính.

## Phần đã tích hợp

Dùng kiến trúc ResNet50 xuất hiện trong notebook làm encoder ảnh vùng người có thêm 20% ngữ cảnh. Trọng số là **ImageNet V2 chính thức của Torchvision**, khác bản pretrained V1 trong notebook. Ảnh BGR chuyển sang RGB, dùng đúng transforms của trọng số, lấy vector 2048 chiều trước lớp phân loại. Encoder được đóng băng; dự án này tự huấn luyện head MIL mới, không gọi đây là checkpoint của tác giả repo.

Hai cấu hình thử: chỉ embedding ResNet50 và embedding nối 201 đặc trưng tư thế/tương tác, tổng 2249 chiều. Cùng tập nguồn 63 video, cùng cửa sổ người và cùng split với V2; 31/40 video train có cửa sổ người đủ số quan sát. Scaler chỉ fit train và cân bằng trọng số từng video. Max-MIL học nhãn cả video, không có nhãn hành vi từng người. Chọn epoch/ngưỡng/cấu hình bằng validation; không thay ngưỡng theo hai video đối chứng.

| Cấu hình | F1 validation | F1 test cũ | AUC test cũ |
|---|---:|---:|---:|
| Baseline | 0.769 | 0.500 | 0.650 |
| resnet_only | 0.800 | 0.545 | 0.750 |
| resnet_pose_interaction | 0.750 | 0.600 | 0.875 |

Cấu hình chọn: **resnet_only**. Cấu hình kết hợp có AUC test cũ cao hơn nhưng không được chọn vì validation kém hơn. Test cũ đã dùng phân tích phát triển, nên đây chỉ là chẩn đoán trên 13 video, không phải kiểm thử độc lập cuối cùng. Không suy ra độ chính xác tại hầm hoặc độ chính xác nhận biết bẻ khóa từ bảng này.

## Đối chứng bãi xe do bạn cung cấp

| Video đối chứng giả định bình thường | Baseline | ResNet50 MIL | Ngưỡng MIL |
|---|---:|---:|---:|
| [Camera Ghi Hình Bãi Giữ Xe Ban Đêm Quán Cháo Cối MT68](https://www.youtube.com/watch?v=a1bcB0_RMiQ) | 0.263 | 0.850 | 0.41 |
| [Lắp Đặt Camera Quan Sát Bãi Giữ Xe  -  KAMERA](https://www.youtube.com/watch?v=5BSCCBLD-_M) | 0.722 | 0.577 | 0.41 |

Nếu tạm coi hai video là bình thường, nhánh mới vẫn vượt ngưỡng ở cả hai. Chúng không được đưa vào train/validation. Kết quả cho thấy cải thiện trên UCF-Crime chưa chuyển thành độ tin cậy vận hành. Video là bãi xe, chưa có nhãn quyền lấy xe và nhãn hành động độc lập. Không tinh chỉnh lại ngưỡng để làm đẹp hai ví dụ này.

## Ba mức hiển thị

| Mức | Điều kiện trong bản thử nghiệm | Cách hiểu |
|---|---|---|
| Xanh lá An toàn | Đủ dữ liệu sơ bộ, chưa hình thành chuỗi dấu hiệu | Chưa ghi nhận, không bảo đảm không có trộm |
| Vàng Nghi vấn | Điểm cao khi ở gần xe đủ lâu, hoặc nhiều dấu hiệu tương tác đồng thời | Cần quan sát và đối chiếu, chưa kết luận |
| Đỏ Nguy hiểm | Điểm cao hơn, đủ pose tin cậy, gần xe ≥4 giây, tay gần vùng xe ≥2 giây, cúi người và chuyển động tay; giữ tín hiệu thêm 2 giây | Proxy nguy cơ cao, cần xác minh; chưa phải bộ nhận dạng bẻ khóa |

Vàng cần tín hiệu giữ 1 giây, đỏ cần 2 giây; hạ mức sau 2 giây để giảm nhấp nháy. Điểm cao đơn lẻ không đủ lên đỏ. Riêng đeo khẩu trang không tham gia quy tắc. Biến động hướng đầu chỉ là proxy 2D yếu, cần đi cùng tiếp cận lặp lại và gần xe mới đóng góp mức vàng. Khi thiếu điểm, theo dõi hoặc chất lượng sẽ hiển thị xám Chưa đủ dữ liệu; đây là trạng thái chất lượng, không phải lớp hành vi thứ tư.

Các ngưỡng đỏ và điều kiện pose là quy tắc kỹ thuật chưa hiệu chỉnh bằng nhãn ba mức. Không có checkpoint hoặc bộ gán nhãn để tuyên bố đang nhận biết chính xác bẻ khóa. Vùng trên bounding box xe chỉ là vùng thay thế, chưa định vị ổ khóa/tay lái. Chủ xe và nhân viên cũng có thể thỏa các dấu hiệu; thông tin cấp quyền và người xác minh vẫn cần thiết.

Mỗi khung hiển thị mức, điểm và lý do trong JSON/bảng web. Tự chụp ảnh ở vàng hoặc đỏ; nâng vàng lên đỏ tạo ảnh mới ngay trong cooldown. Ảnh gốc và ảnh đánh dấu được giữ riêng, sự kiện pending cần người xác minh. Sổ cảnh báo lưu mức tại thời điểm chụp. Metadata ghi điểm mô hình, ngưỡng và chính sách quy tắc; đừng coi ba màu là một mô hình phân lớp ba nhãn đã được huấn luyện.

## Chạy trên web

Tải lại trang, chọn video thật, bật Trích xuất tư thế và tương tác người–xe. Trong Nguồn đặc trưng và điểm đánh giá chọn **ResNet50 theo repo tham khảo (thử nghiệm)**. Bỏ Phát theo thời gian gốc để kiểm tra nhanh. Theo dõi bảng lý do và các mức, không chỉ màu khung. Nếu tiến trình cũ giữ module, khởi động lại Streamlit bằng môi trường .venv.

ResNet50 tải khoảng 98 MB vào runtime/models ở lần đầu; các lần sau dùng cache. File trọng số lớn này không đưa vào Git/ZIP. Head nhỏ, provenance, cache đặc trưng và kết quả đi kèm để tái lập huấn luyện mà không tải lại video/encoder. Muốn chạy suy luận mới cần có trọng số encoder hoặc kết nối mạng lần đầu.

```bash
python scripts/extract_behavior.py
python scripts/extract_visual.py
python scripts/train_behavior.py --visual
python scripts/evaluate_visual.py
python scripts/annotate_video.py data/external_test/tuoitre_ham_chung_cu_20230731.mp4 --behavior --backend resnet_mil --no-evidence --out runtime/annotated/resnet_demo.mp4
python -m pytest -q
```

## Bước cần làm để đánh giá thực tế

Thu thập các hành động hợp lệ dễ nhầm và tình huống nghi vấn dàn dựng có phép trong cùng hầm, gán nhãn người–xe/mốc hành động/quyền di chuyển. Để nhận riêng bẻ khóa, cần video đủ chi tiết vùng tay–khóa và nhãn hành động được xác minh. Chia theo camera/buổi quay trước cắt clip, giữ một test mới khóa riêng; đo bỏ sót, báo giả/camera/giờ, độ trễ và confusion matrix ba mức. Mẫu cột gán nhãn đã thêm risk_level_label và action_confirmed.

## Nguồn

- [Repo đã được yêu cầu kiểm tra](https://github.com/Dilaniu/Theft-Detection-Project-using-OpenCV)
- [Torchvision ResNet50 và trọng số V2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- [COCO class mapping](https://docs.ultralytics.com/datasets/detect/coco/)

LICENSE của repo tham khảo và bản ghi audit được giữ tại artifacts/upstream_review. Toàn bộ phần head mới, tích hợp luồng và quy tắc màu là phần bổ sung của dự án này.
