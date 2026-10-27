# Hướng dẫn demo với video hầm chung cư

Để đọc theo góc nhìn người sử dụng, xem **[Hướng dẫn sử dụng hệ thống](HUONG_DAN_SU_DUNG_HE_THONG.md)**: input, các bước xử lý, output và giải thích từng chỉ số trên web. Tài liệu dưới đây giữ các hướng dẫn demo và xử lý lỗi kỹ thuật.

## Video 240p, người nhỏ hoặc bị che

Chọn model bổ sung YOLO11 và **Cách chạy detector → Người nhỏ / bị che: toàn ảnh 1280 + 4 vùng phóng lớn**. Bấm Phân tích để tạo video mới; video đã xuất giữ nguyên khung cũ. Chế độ phóng vùng chạy chậm hơn và không tái tạo được chi tiết đã mất. Xem [hướng dẫn người nhỏ](PHAT_HIEN_NGUOI_NHO.md). Chờ 4 giây chỉ áp dụng cho điểm cảnh, detector chạy từ đầu.

## Bộ lọc ảnh và tư thế ngồi thấp

Chọn chế độ theo dõi, tại **Tiền xử lý cho phát hiện và tư thế** chọn **Tự động khi tối hoặc tương phản thấp**; có thể bật **Lọc nhiễu nhẹ (bilateral)**. Giữ bật trích xuất tư thế. Bảng kết quả có thêm ngồi thấp/co gối ước lượng, góc gối và cờ đủ dữ liệu. Thiếu tư thế tin cậy hiển thị xám thay vì xanh. Xem [cách hoạt động, giới hạn và lệnh chạy đối chiếu](BO_LOC_VA_TU_THE.md). Tư thế ngồi thấp chỉ phối hợp với thao tác tay–xe kéo dài, không tự kết luận trộm; hiện chưa có detector khẩu trang/mũ chuyên dụng.

## Thử model trong thư mục model

Chọn **Theo dõi trực tiếp trên video**, tại **Model phát hiện người và xe** chọn **Model bổ sung: yolo_behavior_model.pt (YOLO11 COCO)**. Chọn video, giữ nhánh chấm điểm hành vi mong muốn rồi bấm **Phân tích**. Video kết quả, ID theo dõi, ba màu và chụp ảnh tự động đều sử dụng detector đã chọn. JSON tải về ghi tên checkpoint và SHA-256 trong `detector_metadata` để biết chính xác model nào đã chạy.

Để so sánh, chạy lại cùng video với **YOLOv8n hiện tại**, giữ nguyên các tùy chọn hành vi và nguồn điểm. Không dùng số lượng khung hoặc điểm tin cậy người để kết luận độ chính xác phát hiện trộm.

Checkpoint bổ sung hiện khai báo 80 lớp COCO, tác vụ `detect`, có `person` và `motorcycle`, chưa có lớp `theft`/`lock-breaking`. Metadata train lưu đường dẫn `coco.yaml`. Do đó tích hợp này thử thay bộ phát hiện đối tượng, không thay bộ phân lớp hành vi. Xem [chi tiết tích hợp](TICH_HOP_MODEL_BO_SUNG.md).

Chạy bằng Terminal:

```bash
python scripts/annotate_video.py data/external_test/youtube_5BSCCBLD-_M.mp4 --detector-weights model/yolo_behavior_model.pt --behavior --backend scene --out runtime/annotated/custom_model.mp4
```

Nếu web đang chạy phiên bản cũ, dừng bằng `Ctrl+C` rồi chạy lại Streamlit. Lựa chọn detector chỉ có trong chế độ theo dõi; phân tích tổng quan vẫn dùng baseline cấp video.

## Bản mới ResNet50 và ba mức xanh vàng đỏ

Tải lại web, chọn video thật, bật **Trích xuất tư thế và tương tác người–xe**. Tại **Nguồn đặc trưng và điểm đánh giá**, chọn **ResNet50 theo repo tham khảo (thử nghiệm)** để thử nhánh mới. Có thể giữ Baseline toàn cảnh hoặc chọn MIL tư thế để đối chiếu. Lần đầu ResNet50 tải khoảng 98 MB trọng số chính thức; các lần sau dùng cache trên máy. [Báo cáo tích hợp và kết quả](TICH_HOP_DILANIU.md).

- **Xanh lá An toàn:** đủ dữ liệu sơ bộ và chưa ghi nhận chuỗi dấu hiệu cần xem xét; không bảo đảm không có trộm.
- **Vàng Nghi vấn:** điểm và tương tác cần quan sát, hoặc nhiều dấu hiệu cục bộ kết hợp đủ thời gian.
- **Đỏ Nguy hiểm:** điểm cao hơn và chuỗi thao tác tay gần xe, cúi người, chuyển động tay kéo dài; là nguy cơ cao cần xác minh, chưa nhận dạng được riêng hành động bẻ khóa.

Khẩu trang hoặc nhìn quanh đơn lẻ không làm tăng mức. Proxy đổi hướng đầu chỉ đóng góp cùng tiếp cận lặp lại và ở gần xe. Vàng giữ tín hiệu 1 giây, đỏ 2 giây; hạ mức sau 2 giây. Xám là trạng thái chưa đủ điểm/theo dõi/chất lượng, không phải một lớp hành vi thứ tư. Bảng web và JSON ghi mức cùng lý do. Ba mức hiện là lớp quy tắc, chưa phải bộ phân lớp ba nhãn có kiểm định độc lập.

## Tự chụp ảnh tại thời điểm có dấu hiệu nghi vấn

Trong **Theo dõi trực tiếp trên video**, giữ bật **Tự chụp ảnh khi xuất hiện khung vàng hoặc đỏ**. Khi quy tắc đang chọn đánh dấu một người cần xem xét, hệ thống tự lưu và tạo cảnh báo pending trong Sổ cảnh báo. Không cần bấm lưu thủ công.

Mỗi lần ghi nhận gồm:

- `original.png`: đúng frame nguồn đã giải mã, giữ độ phân giải nguồn và chưa vẽ lên ảnh; không lấy frame đầu video thay cho frame nghi vấn.
- `annotated.jpg`: ảnh có khung người, điểm đánh giá và thời gian để xem nhanh.
- `metadata.json`: camera, tên/hash video, chỉ số frame từ 0, thời gian trong video, ID theo dõi, tọa độ khung trên ảnh nguồn, nguồn điểm, điểm/ngưỡng và hash các ảnh.

Các file nằm tại `runtime/evidence/<run_id>/<evidence_id>/`. PNG giữ nguyên pixel đã giải mã; file video gốc vẫn được giữ riêng. Hash phục vụ kiểm tra tệp có thay đổi, không biến kết quả mô hình thành kết luận đã xác minh.

Các khung dương liên tiếp của cùng ID được gộp thành một đợt, chụp ảnh đầu đợt; khoảng ngắt tối đa 3 giây vẫn được gộp. Hai ảnh cùng mức của cùng ID cách nhau tối thiểu 10 giây; riêng nâng vàng lên đỏ chụp ảnh mới ngay để giữ đúng thời điểm tăng mức. Nếu nhiều người cùng được đánh dấu lần đầu trong một frame, dùng chung ảnh và ghi đủ ID. Metadata lưu mức tại lúc chụp, thời điểm quan sát cuối và điểm cao nhất của đợt; ảnh đã chụp giữ nguyên.

Sau khi chạy xong có mục **Ảnh ghi nhận tự động** để xem/tải ảnh. Có thể mở **Sổ cảnh báo**, chọn mã sự kiện để xem ảnh, tải PNG/JPG/JSON và xác minh confirmed hoặc false_alarm. Ảnh đã ghi vẫn được giữ nếu bấm Dừng xử lý giữa chừng. Chạy lại cùng video tạo một lượt phân tích mới, có run_id riêng.

Nếu không có khung vàng hoặc đỏ thì không tự tạo ảnh nghi vấn. Chế độ Phân tích tổng quan chỉ chấm điểm cả video nên không tự xác định người/frame cần chụp. Mọi cảnh báo tự tạo ban đầu là pending, chưa xác nhận người trong ảnh đang trộm xe. Các file runtime và video cá nhân không được đưa lên Git.

## Lỗi unexpected keyword argument behavior

Thông báo `stream_video() got an unexpected keyword argument 'behavior'` xuất hiện khi giao diện mới gọi tham số hành vi nhưng tiến trình Streamlit còn giữ hàm của phiên bản cũ trong RAM. Mã mới tự kiểm tra chữ ký hàm và nạp lại module khi phát hiện lệch phiên bản; lỗi khởi tạo cũng được đưa vào phần xử lý lỗi của giao diện.

Sau khi cập nhật source, tải lại trang và bấm Phân tích. Nếu phiên đang mở vẫn giữ mã cũ, dừng server trong Terminal bằng Ctrl+C rồi chạy:

```bash
cd /Users/mynameisfuong/Documents/practice_ck
source .venv/bin/activate
python -m streamlit run app.py --server.port 8502
```

Mở `http://localhost:8502`. Lỗi này không yêu cầu huấn luyện lại mô hình hoặc cài lại thư viện. Khi chuyển dự án sang máy khác, cần cập nhật đồng bộ app.py và toàn bộ thư mục src.

## Bổ sung đặc trưng hành vi và đối chứng bình thường

Phiên bản mới có checkbox **Trích xuất tư thế và tương tác người–xe**. Khi bật, ứng dụng hiển thị khung xương và các đặc trưng: vị trí/góc khớp, chuyển động cổ tay, tay gần vùng xe, thời gian tương tác và chuyển động người–xe. Tốc độ hiển thị mục tiêu là 4 FPS, đặc trưng cập nhật 2 lần/giây. Tỷ lệ khớp qua ngưỡng dựa trên confidence của pose model, không phải xác minh nhìn rõ bằng người gán nhãn.

Chọn **MIL tư thế (thử nghiệm)** trong Nguồn đặc trưng và điểm đánh giá để nghiên cứu head V2 học bằng nhãn cả video. Head V2 chưa vượt baseline trên validation và còn báo nhầm trên hai video bãi xe bạn cung cấp. Điểm MIL không phải xác suất người đó ăn trộm. [Báo cáo V2](NANG_CAP_HANH_VI.md) mô tả thí nghiệm trước nhánh ResNet50.

Hai video mới đã tải vào `data/external_test` và xuất hiện trong danh sách chọn:

- `youtube_a1bcB0_RMiQ.mp4`: bãi xe ban đêm quán Cháo Cối MT68, khoảng 4 phút 52 giây.
- `youtube_5BSCCBLD-_M.mp4`: video camera bãi giữ xe KAMERA, khoảng 59 giây.

Hai video được giữ ngoài train/validation. Khi dùng nhãn bình thường giả định, MIL hiện báo nhầm; không đổi ngưỡng chỉ để làm đẹp kết quả. Cần gán nhãn hành động và xác minh quyền lấy xe để đánh giá thực tế. Mẫu cột gán nhãn ở `data/action_annotations_template.csv`.

## Mới Xem video và khung người cập nhật theo thời gian

Đã sửa lỗi video đen bằng cách chuyển bản phát sang H.264. Sau khi chọn hoặc upload video, phần **Video nguồn** xuất hiện ngay, chưa cần bấm phân tích.

1. Tải lại trang đang mở, ví dụ `http://localhost:8502` (hoặc 8501 tùy Terminal).
2. Nếu đang có file upload hình vẽ, bấm nút xóa file đó trước khi chọn video khác trong danh sách. File upload được ưu tiên hơn file chọn.
3. Chọn `tuoitre_ham_chung_cu_20230731.mp4` để thử khung người trong bãi xe; có thể bấm Play để xem video gốc.
4. Chọn **Theo dõi trực tiếp trên video**. Giữ **Phát theo thời gian gốc** để xem cập nhật theo thời gian video; bỏ chọn nếu muốn xử lý nhanh theo sức máy.
5. Bấm **Phân tích**. Vùng **Đang theo dõi và chấm điểm** hiển thị frame đang xử lý với khung người, ID tạm thời và điểm. Chờ đủ 4 giây đầu để có điểm cảnh, sau đó điểm cập nhật mỗi 4 giây.
6. Khi xử lý xong, xem phần **Phát lại video có khung và điểm theo thời gian** hoặc tải video/JSON. Nút **Dừng xử lý** hủy lần chạy hiện tại; chỉ lần chạy hoàn tất mới có video đầu ra đầy đủ.

Màu khung dùng chính sách ba mức ở đầu hướng dẫn. `det` là độ tin cậy phát hiện người; `scene` là điểm toàn cảnh; điểm MIL/ResNet50 là điểm thử nghiệm từ nhãn video. Người đang lấy xe hợp lệ cũng có thể thỏa dấu hiệu nghi vấn. Không hạ ngưỡng để ép các video hiện màu đỏ.

Detector YOLOv8n/ByteTrack đã được cài và trọng số đã có trên máy hiện tại. Trên máy mới, chạy thêm sau khi cài requirements:

```bash
python scripts/setup_detector.py
```

Nếu Terminal Streamlit đang dùng môi trường cũ và báo thiếu module, dừng bằng Ctrl+C, kích hoạt `.venv`, chạy `python -m pip install -r requirements.txt`, rồi mở lại ứng dụng. Video kết quả được lưu dưới `runtime/annotated`, không có âm thanh. FPS xử lý hiển thị trên giao diện, có thể thấp hơn tốc độ nguồn trên máy chậm. Chế độ này đọc file CFR theo thời gian; chưa kết nối camera RTSP.

Các mục kết quả và lệnh `predict.py` phía dưới mô tả chế độ **Phân tích tổng quan** ban đầu. Chọn đúng chế độ này khi muốn tái lập các điểm 0,254 và 0,423. Điểm cảnh theo ngữ cảnh trượt của chế độ trực tiếp có thể khác điểm gộp cả video.

## Video đã tìm và tải vào máy

Hai file sau chỉ dùng kiểm tra bên ngoài, chưa dùng huấn luyện hoặc chọn ngưỡng. Nguồn và SHA-256 được ghi tại `data/external_test/sources.json`.

| File | Nội dung và nguồn | Thời lượng |
|---|---|---:|
| `data/external_test/tuoitre_ham_chung_cu_20230731.mp4` | Tuổi Trẻ ngày 31/07/2023, [Toàn bộ diễn biến nghi phạm đột nhập hầm chung cư trộm xe máy](https://tuoitre.vn/video/toan-bo-dien-bien-nghi-pham-dot-nhap-ham-chung-cu-trom-xe-may-146486.htm). Có cảnh người tương tác với xe trong khu đỗ xe; video có cắt cảnh, phóng hình và logo. | 44,64 giây |
| `data/external_test/voh_orchard_parkview_20191008.mp4` | VOH ngày 08/10/2019, [Truy bắt kẻ trộm xe máy vượt barie tẩu thoát ở chung cư TPHCM](https://voh.com.vn/phap-luat/truy-bat-ke-trom-xe-may-vuot-barie-tau-thoat-o-chung-cu-tphcm-337453.html). Bài ghi nguồn clip SGGP. Video tập trung vào cổng/barie, không cho thấy đầy đủ quá trình tác động vào khóa xe. | 29,83 giây |

Đây là video báo chí có bản quyền, không phải dataset mở đã được cấp phép lại. Giữ nguồn khi đưa vào phần minh họa; không đóng gói chúng như dữ liệu do nhóm tự quay. Chúng không được tự động đưa vào các tập train/validation/test hiện có.

## Chạy giao diện trên máy hiện tại

Nếu server đang chạy, mở `http://localhost:8501`. Nếu giao diện đang báo thay đổi file, bấm Rerun hoặc tải lại trang để hiện hai video mới.

Nếu server đã dừng, mở Terminal và chạy:

```bash
cd /Users/mynameisfuong/Documents/practice_ck
source .venv/bin/activate
python -m streamlit run app.py
```

Giữ Terminal mở trong lúc demo. Bấm Ctrl+C để dừng server. Nếu báo cổng 8501 đang được dùng, kiểm tra trang đang chạy trước; hoặc chạy `python -m streamlit run app.py --server.port 8502` và mở `http://localhost:8502`.

Nếu chạy từ bản ZIP trên máy khác, cần Python 3.12–3.13 và cài môi trường một lần:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Trên Windows thay lệnh source bằng `.venv\Scripts\Activate.ps1`. File video báo chí chỉ nằm trong workspace hiện tại, không có trong ZIP ban đầu; hãy dùng file có sẵn ở máy hoặc video nhóm chuẩn bị và tải lên giao diện.

## Các bước thao tác demo

1. Chọn vai trò **Bảo vệ**. Vai trò chỉ mô phỏng giao diện, không phải đăng nhập bảo mật.
2. Vào tab **Phân tích video**.
3. Chọn chế độ **Phân tích tổng quan**. Trong **Hoặc video đã tải về**, chọn một trong hai tên file ở bảng trên. Nếu dùng file khác, bấm **Upload**. Nếu đồng thời tải file lên và chọn một file trong danh sách, ứng dụng ưu tiên file tải lên.
4. Bật **Phân tích liên tục các cửa sổ 4 giây** để xem đường điểm theo thời gian. Đây vẫn là phân tích một file sau khi tải lên, chưa phải giám sát camera trực tiếp.
5. Bấm **Phân tích** và đợi kết quả.
6. Xem điểm video, ngưỡng, trạng thái chất lượng, video và bảng cửa sổ. Bấm **Tải kết quả JSON** để lưu kết quả cho báo cáo.
7. Nếu quyết định là **CẦN BẢO VỆ XEM XÉT**, sẽ có nút **Lưu cảnh báo chờ xác minh**. Sau khi lưu, vào **Sổ cảnh báo** để đánh dấu pending, confirmed hoặc false_alarm và nhập ghi chú.

Nút lưu cảnh báo tự xuất hiện khi điểm video đạt ngưỡng và chất lượng được chấp nhận. Hai video báo chí hiện không đạt ngưỡng nên không xuất hiện nút đó. Để trình diễn riêng luồng lưu/xác minh, có thể chọn `loitering_then_move.mp4` trong nhóm video minh họa. Đây là hình vẽ tổng hợp chỉ thử phần mềm; không dùng nó làm bằng chứng mô hình phát hiện đúng trộm thật.

## Kết quả đã chạy với mô hình hiện tại

Mô hình ExtraTrees, ngưỡng từ validation 0,50, chưa huấn luyện lại trong bước tìm video này.

| Video | Điểm cấp video | Quyết định | Thời gian xử lý ghi nhận |
|---|---:|---|---:|
| Tuổi Trẻ | 0,254 | low_score | 8,58 giây |
| VOH | 0,423 | low_score | 2,65 giây |

Cả hai qua kiểm tra chất lượng sơ bộ nhưng đều không tạo cảnh báo cấp video. Với nội dung sự kiện theo bài báo, đây là các trường hợp bỏ sót trong thử nghiệm bên ngoài; hai mẫu không đủ để tính độ chính xác đại diện.

File VOH có một khoảng điểm ngữ cảnh vượt ngưỡng từ 0–8 giây trong JSON dense, nhưng điểm cả video vẫn thấp. Đường điểm là thử nghiệm dùng ngữ cảnh trượt, chưa được kiểm định định vị và không được coi là phát hiện chính xác thời điểm vượt barie. Giao diện hiện quyết định lưu cảnh báo theo điểm cả video.

Không đổi ngưỡng sau khi nhìn hai kết quả để làm demo đẹp hơn. Khi nâng cấp mô hình, có thể dùng hai video này để phân tích lỗi; nếu đã dùng để điều chỉnh, phải chuẩn bị video khác làm tập kiểm thử cuối.

Kết quả chi tiết:

- `artifacts/test_tuoitre_ham.json`
- `artifacts/test_voh_orchard.json`

## Chạy bằng dòng lệnh

```bash
python scripts/predict.py data/external_test/tuoitre_ham_chung_cu_20230731.mp4 --dense --out artifacts/test_tuoitre_ham.json
python scripts/predict.py data/external_test/voh_orchard_parkview_20191008.mp4 --dense --out artifacts/test_voh_orchard.json
```

Không cần chạy `train.py` để test vì đã có `artifacts/model.joblib`. Lệnh train sẽ ghi đè mô hình hiện tại.

## Đọc đầu ra đúng cách

- `video_score`: điểm từ bộ phân lớp, chưa hiệu chuẩn thành xác suất phạm tội.
- `threshold`: ngưỡng đã chọn bằng tập validation hiện có.
- `review`: điểm đủ cao và qua kiểm tra chất lượng, cần người xem xét.
- `low_score`: điểm thấp, không bảo đảm video không có trộm.
- `insufficient_quality`: tín hiệu không đạt kiểm tra chất lượng sơ bộ.
- `windows`: thống kê chất lượng và điểm ngữ cảnh; chưa có đánh giá chính xác mốc thời gian hành vi.

Để đánh giá đầy đủ, test thêm video chủ xe lấy xe, nhân viên dắt xe và người đi ngang ở cùng môi trường. Ghi số báo giả và bỏ sót, không chỉ trình diễn các video dự đoán đúng.
