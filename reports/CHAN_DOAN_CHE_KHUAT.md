# Chẩn đoán hai khung hình người bị xe che khuất

Nhật ký được kiểm tra: `runtime/annotated/544226dc38c0a21638ba7f7b_1793094061944150000.json`, nguồn upload `cd89cbf55c51cf25.mp4`. Lần chạy dùng YOLOv8n, nguồn điểm `scene`, bilateral và làm nét; không dùng checkpoint bổ sung trong thư mục `model`.

| Thông tin | 16,76 giây | 29,52 giây |
|---|---|---|
| Phát hiện người | ID 38, confidence 0,663 | Không có người được trả về |
| Xe máy | 5 phát hiện | 1 phát hiện |
| Điểm cảnh / ngưỡng vàng | 0,350 / 0,50 | 0,359 / 0,50 |
| Điểm pose MIL hiển thị | 0,166, không phải nguồn điểm được chọn | Không có |
| Thời gian gần xe | 2,76 giây | Không đánh giá được |
| Khớp thân/tay/chân qua confidence 0,4 | 50% | Không đánh giá được |
| Đủ khớp chân tính co gối | Không | Không đánh giá được |
| Tay gần vùng xe theo ước lượng | Không ghi nhận được | Không đánh giá được |
| Chuyển động đầu đạt điều kiện proxy | Không | Không đánh giá được |

Điểm hai gối khoảng 0,267 và 0,092, hai cổ chân khoảng 0,065 và 0,029; đều dưới 0,4. Vì vậy chưa đo được tư thế ngồi xổm từ mô hình. Một cổ tay cũng dưới ngưỡng. Điều này mô tả **quan sát của mô hình**, không chứng minh người không ngồi xổm hoặc không thao tác với xe.

Nhìn quanh được ước lượng bằng biến động vị trí mũi so với vai qua chuỗi thời gian, không phải model hướng nhìn; việc con người thấy quay đầu không đảm bảo tín hiệu 2D đạt điều kiện. Khẩu trang/mũ chưa có detector chuyên dụng. Chỉ từ phụ kiện và tư thế không xác minh được ý định hay quyền sở hữu xe.

Ở 29,52 giây, lỗi nằm trước bước phân loại: detector không trả về người. Hạ ngưỡng điểm cảnh không sửa được ca bỏ sót này. Xe che phần thân dưới, góc nhìn, kích thước và tư thế khác dữ liệu train có thể góp phần; nhật ký chưa xác định riêng tác động của từng yếu tố.

## Điều chỉnh đã thực hiện

Ở chế độ tư thế, người chỉ có quan sát một phần sẽ không hiển thị xanh nếu chưa có chuỗi tương tác đủ để cảnh báo và còn thiếu khớp hai chân hoặc cổ tay. Trường hợp này hiển thị xám, kèm lý do. Các tương tác tay–xe đủ tin cậy vẫn có thể lên vàng/đỏ ngay cả khi chân bị che; không bắt buộc chân phải lộ mới cảnh báo bằng nhánh thao tác tay.

Thêm cột **Điều kiện còn thiếu / giới hạn** để thấy điểm thấp, chưa đủ thời gian gần xe, thiếu khớp, chưa quan sát tay gần xe hoặc proxy đầu chưa đạt điều kiện. Đây là danh sách điều kiện quan sát, không phải độ quan trọng đặc trưng học được hay kết luận “không trộm”. Khi không phát hiện người, web thông báo rõ chưa thể đánh giá hành vi.

Đã chạy lại chính sách mới trên phát hiện/tư thế đã lưu: 16,76 giây chuyển từ xanh sang xám; 29,52 giây vẫn không có người được phát hiện. [JSON kiểm chứng](../artifacts/occlusion_diagnosis.json). Đây là replay chính sách trên dữ liệu có sẵn, chưa phải detector mới hay bằng chứng tăng độ chính xác phát hiện hành vi.

Khởi động lại Streamlit và bấm **Phân tích** để tạo video mới. Video đã xuất trước đó giữ màu cũ. Cần cải thiện riêng detector với dữ liệu người bị che khuất, và huấn luyện mô hình hành động bằng clip có nhãn thao tác thực tế cùng các hoạt động bình thường tương tự; sửa hiển thị thiếu dữ liệu chưa giải quyết được bỏ sót người.
