# Bộ lọc ảnh và bổ sung tư thế ngồi thấp

## Cách thử trên web

1. Khởi động lại Streamlit, tải video thật và chọn **Theo dõi trực tiếp trên video**.
2. Chọn detector muốn thử, kể cả checkpoint trong thư mục `model`.
3. Ở **Tiền xử lý cho phát hiện và tư thế**, chọn **Tự động khi tối hoặc tương phản thấp**. Có thể bật **Lọc nhiễu nhẹ (bilateral)**; chỉ bật **Làm nét nhẹ** để đối chiếu khi cần.
4. Giữ bật **Trích xuất tư thế và tương tác người–xe**, chọn cùng nguồn điểm qua các lần thử rồi bấm **Phân tích**.
5. Quan sát bảng: **Ngồi thấp/co gối (ước lượng)**, **Góc gối trung bình (độ)**, **Tay gần vùng xe**, **Chuyển động tay**, **Đủ dữ liệu đánh giá** và **Lý do**. Ô tư thế trống có nghĩa không đủ điểm chân tin cậy, không phải đang đứng.
6. Đối chiếu cùng video bằng **Giữ ảnh gốc**, tắt cả lọc nhiễu và làm nét. Tải hai video và JSON để kiểm tra khung bị bỏ sót, ID đổi, cảnh báo nhầm và sự kiện bị bỏ qua.

Bộ lọc và tư thế mới chỉ áp dụng trong chế độ theo dõi. Chế độ tổng quan vẫn dùng mô hình baseline đã huấn luyện.

## Các bước xử lý ảnh

| Tùy chọn | Xử lý thực hiện | Giới hạn |
|---|---|---|
| Giữ ảnh gốc | Không đổi màu/độ sáng nếu không bật hai bộ lọc riêng | Mốc đối chiếu |
| Tự động | CLAHE trên kênh sáng LAB khi độ sáng xám trung bình <65 hoặc độ lệch chuẩn <30; thêm gamma 0,75 khi tối | Các mốc là quy tắc thực nghiệm trên thang 0–255, chưa hiệu chỉnh theo camera |
| Tăng tương phản cho video thiếu sáng | Luôn CLAHE; thêm gamma khi độ sáng trung bình <65 | Có thể làm lộ nhiễu hoặc mất tương phản ở cảnh vốn sáng |
| Lọc nhiễu nhẹ | Bilateral kích thước 5, sigma màu/không gian 25 | Không loại bỏ được mọi nhiễu; có thể mất chi tiết nhỏ |
| Làm nét nhẹ | Unsharp: 1,25 lần ảnh trừ 0,25 lần Gaussian sigma 1 | Không tái tạo được ổ khóa hoặc bàn tay vốn bị nhòe |

Thứ tự: tăng tương phản → gamma nếu cần → lọc nhiễu → làm nét. Không thay đổi kích thước hoặc hình học so với ảnh đưa vào detector. Mỗi frame JSON có trường `preprocessing` ghi độ sáng, tương phản và danh sách bộ lọc thực sự áp dụng; web cũng hiển thị danh sách này.

Ảnh đã lọc chỉ đưa vào **detector và mô hình tư thế**. Các đặc trưng màu/chuyển động của baseline và ảnh crop ResNet vẫn lấy từ ảnh chưa lọc để hạn chế lệch phân phối đầu vào của mô hình đã train. Pose MIL nhận điểm khớp mới nên điểm vẫn có thể thay đổi. Kiểm tra chất lượng và ảnh gốc PNG lưu bằng chứng tiếp tục dùng ảnh nguồn; hệ thống không dùng ảnh làm sáng để giả vờ video gốc đã đủ chất lượng. Video đánh dấu hiển thị trên nền ảnh nguồn, không phải nền ảnh tăng sáng.

## Tín hiệu tư thế mới

Mô hình YOLO Pose hiện có 17 điểm cơ thể. Phần mới tính góc hông–gối–cổ chân của hai chân. Chỉ chấp nhận khi cả 6 điểm tin cậy ≥0,4, tọa độ hữu hạn và các đoạn chân dài ít nhất 3 pixel.

`crouching_proxy=True` khi hai góc gối ≤115° và độ chênh dọc hông–gối / chiều dài đùi ≤0,65 ở cả hai chân. Đây là **ước lượng tư thế co gối/ngồi thấp bằng hình học 2D**, có thể nhầm với ngồi ghế hoặc bị ảnh hưởng bởi góc quay; chưa phải mô hình được train riêng để phân biệt ngồi xổm, sửa xe và bẻ khóa. Mất một điểm chân sẽ trả `lower_body_valid=False`, không tự suy đoán điểm bị che.

Các trường mới nằm ngoài vector đặc trưng cũ. Head MIL/ResNet đã train vẫn giữ nguyên số chiều và trọng số; thay đổi này bổ sung tín hiệu cho quy tắc hiển thị, không phải đã huấn luyện lại bộ phân lớp hành vi.

## Đánh giá mức và vấn đề nhiều khung xanh

Cập nhật xử lý che khuất: tỷ lệ khớp ≥0,5 chưa có nghĩa nhìn thấy chân hoặc tay. Nếu chưa có chuỗi tương tác cảnh báo mà còn thiếu khớp hai chân/cổ tay, hệ thống giữ màu xám thay vì xanh. Các tương tác tay đủ tin cậy vẫn có thể cảnh báo khi chân bị che. Xem [chẩn đoán hai khung hình thực tế](CHAN_DOAN_CHE_KHUAT.md).

Trước đây chế độ cảnh có thể trả xanh khi điểm cảnh thấp dù tư thế đang thiếu. Nay **khi bật phân tích tư thế**, phải có quan sát tư thế mới trong 0,8 giây, ≥4 lần quan sát và tỷ lệ khớp tin cậy ≥0,5 mới đủ đánh giá. Nếu thiếu sẽ xám **Chưa đủ dữ liệu**. Vẫn phải có điểm của nhánh đang chọn, chất lượng cảnh và ID theo dõi tin cậy. Khi tắt tư thế, baseline vẫn đánh giá bằng điểm cảnh và gần xe như trước.

Không hạ ngưỡng chỉ để có nhiều khung đỏ hơn. Các tín hiệu mới phối hợp như sau:

- **Vàng:** có thêm lý do ngồi thấp nếu gần xe ≥3 giây, tay gần vùng xe ≥2 giây, có chuyển động tay tương đối hợp lệ và tư thế co gối. Chuỗi dấu hiệu phải giữ ≥1 giây. Các quy tắc vàng trước đây vẫn được giữ, nên một lý do có thể trùng thời điểm với lý do tay/chuyển động.
- **Đỏ:** ngồi thấp có thể thay điều kiện cúi người; vẫn cần gần xe ≥4 giây, tay gần xe ≥2 giây, chuyển động tay, tư thế tin cậy, điểm mô hình vượt ngưỡng đỏ và chuỗi điều kiện giữ ≥2 giây.
- **Ngồi xổm riêng lẻ, khẩu trang hoặc mũ:** không làm tăng mức. Chủ xe mở cốp, kiểm tra lốp hoặc sửa xe vẫn có thể tạo tương tác tương tự, cần xác minh.

Khẩu trang/mũ **chưa được phát hiện tự động**: checkpoint người dùng có 80 lớp COCO, mô hình tư thế có 17 điểm khớp, cả hai đều chưa có nhãn phụ kiện tương ứng. Không suy ra khẩu trang/mũ từ việc thiếu điểm mặt. Muốn bổ sung cần model phụ kiện có trọng số và nhãn phù hợp, dữ liệu hầm và tập kiểm thử; ngay cả khi có cũng chỉ dùng làm mô tả quan sát, không chứng minh hành vi trộm.

## Kiểm chứng và cách đối chiếu

Các kiểm thử bao gồm: không sửa ảnh gốc, giữ nguyên hình học, tự động không lọc ảnh đủ sáng/tương phản, chân thẳng/co gối/che khớp, tư thế thiếu hoặc quá cũ phải xám, ngồi thấp/mũ/khẩu trang riêng lẻ không cảnh báo, tương tác kéo dài mới chuyển vàng/đỏ và chụp đúng frame vẫn hoạt động.

Chạy đối chiếu video thật bằng CLI:

```bash
python scripts/compare_preprocessing.py data/external_test/tuoitre_ham_chung_cu_20230731.mp4 --detector-weights model/yolo_behavior_model.pt --denoise --out-dir runtime/filter_comparison/theft_source
```

Lệnh chạy hai lần `off` và `auto`, bật tư thế, giữ nguồn điểm `scene`, không tạo bản ghi trong sổ cảnh báo. Video và nhật ký từng lần nằm trong thư mục output; `comparison.json` tổng hợp số quan sát theo từng mức, số frame lọc và thời gian xử lý. Các số đếm là **lượt người–frame**, không phải số người duy nhất, số vụ trộm hay accuracy.

Chưa có nhãn tư thế/hành động đối chứng cho video hầm, vì vậy chưa thể khẳng định tăng precision/recall. Cần đánh dấu người, thời gian xảy ra hành động và các video bình thường tương tự để đo cả bỏ sót lẫn báo nhầm. Nếu vẫn ít cảnh báo, kiểm tra lần lượt detector có thấy người/xe không, các khớp có đủ không, khoảng cách/tay có quan sát được không, chất lượng nguồn và điểm mô hình; không xem mọi khung xanh là bằng chứng mô hình đã nhận đúng.

## Kết quả chạy đối chiếu trên máy

Cùng detector `model/yolo_behavior_model.pt`, bật tư thế, dùng điểm `scene`. Nhánh tự động kèm lọc bilateral; nhánh ảnh gốc tắt cả bộ lọc. Cả bốn video output đều mở đọc lại được. [JSON kiểm chứng](../artifacts/preprocessing_pose_verification.json) ghi cấu hình, hash detector, thời gian và các số đếm chi tiết.

| Video | Bộ lọc | Xanh | Vàng | Đỏ | Xám |
|---|---|---:|---:|---:|---:|
| Tuổi Trẻ, nguồn sự kiện 44,64 giây | Tắt | 31 | 1 | 0 | 25 |
| Tuổi Trẻ, nguồn sự kiện 44,64 giây | Tự động + bilateral | 29 | 0 | 0 | 26 |
| `youtube_5BSCCBLD-_M`, đối chứng tạm coi bình thường 59,03 giây | Tắt | 92 | 155 | 0 | 138 |
| `youtube_5BSCCBLD-_M`, đối chứng tạm coi bình thường 59,03 giây | Tự động + bilateral | 96 | 153 | 0 | 144 |

Số đếm là lượt người–frame nên các hàng không có cùng tổng: bộ lọc có thể làm đổi kết quả phát hiện. Đây là tín hiệu cần thận trọng: video đối chứng vẫn có nhiều lượt vàng cần kiểm tra báo nhầm, trong khi video nguồn sự kiện có rất ít vàng và không đỏ. **Chưa chứng minh cải thiện nhận dạng hành vi**; không nên bật lọc hoặc đổi ngưỡng chỉ để tăng màu đỏ. Tập video này chưa có nhãn người và thời gian hành động để tính precision/recall. Bộ lọc được giữ là tùy chọn, mặc định ảnh gốc.
