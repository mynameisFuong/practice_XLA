# Hướng dẫn sử dụng hệ thống cảnh báo nghi vấn trộm xe máy

Tài liệu này hướng dẫn đưa video vào hệ thống, lựa chọn cách phân tích, đọc các chỉ số trên web và lấy kết quả ra để xem xét. Các tên nút và trường bên dưới tương ứng với giao diện hiện tại.

Hệ thống nhận **file video**, phát hiện và theo dõi người–xe, tính điểm đánh giá rồi hiển thị dấu hiệu cần xem xét. Kết quả hỗ trợ bảo vệ kiểm tra; bản hiện tại chưa được kiểm định để kết luận một người đang trộm xe hoặc đang bẻ khóa.

**Model bổ sung trong thư mục `model`:** Trong chế độ theo dõi, chọn **Model phát hiện người và xe → Model bổ sung: yolo_behavior_model.pt (YOLO11 COCO)** để thử checkpoint mới. Model này phát hiện đối tượng người/xe; nhánh chấm điểm hành vi được chọn riêng bên dưới. Chạy lại cùng video với **YOLOv8n hiện tại** để đối chiếu. Tên detector thực sự đã dùng xuất hiện ở kết quả và JSON. Xem [hướng dẫn thử model bổ sung](TICH_HOP_MODEL_BO_SUNG.md).

**Bộ lọc ảnh và tư thế ngồi thấp:** Chọn **Tiền xử lý cho phát hiện và tư thế → Tự động khi tối hoặc tương phản thấp**, có thể bật lọc nhiễu nhẹ. Giữ bật phân tích tư thế để xem các cột co gối, góc gối, thao tác tay và lý do. Khi thiếu tư thế tin cậy, chế độ này hiển thị xám thay vì xanh. Khẩu trang/mũ chưa có detector chuyên dụng và không được dùng riêng để tăng cảnh báo. Xem [hướng dẫn bộ lọc và tư thế](BO_LOC_VA_TU_THE.md).

**Video người nhỏ/240p:** chọn model bổ sung YOLO11 và **Cách chạy detector → Người nhỏ / bị che: toàn ảnh 1280 + 4 vùng phóng lớn**, rồi bấm Phân tích. Chế độ này chạy chậm hơn và có thể tăng phát hiện nhầm; ứng viên yếu vẫn xám. Xem [hướng dẫn và kết quả thử](PHAT_HIEN_NGUOI_NHO.md).

## 1 Mở ứng dụng

Trên máy đang có dự án và môi trường đã cài, mở Terminal:

```bash
cd /Users/mynameisfuong/Documents/practice_ck
source .venv/bin/activate
python -m streamlit run app.py --server.port 8502
```

Mở [http://localhost:8502](http://localhost:8502). Nếu Terminal thông báo cổng khác, mở địa chỉ được in trong Terminal. Giữ Terminal chạy trong lúc sử dụng; nhấn `Ctrl+C` để dừng server.

Không cần huấn luyện lại mô hình để phân tích video. Nếu chạy trên máy mới, xem phần cài đặt trong [README](../README.md).

Ở thanh bên trái, chọn **Bảo vệ** hoặc **Quản trị** để sử dụng chức năng phân tích. Vai trò **Cư dân** mới là giao diện mô phỏng. Việc chọn vai trò chưa phải đăng nhập hay phân quyền bảo mật.

## 2 Đưa input vào hệ thống

### 2.1 Input cần chuẩn bị

| Input | Cách cung cấp | Hệ thống sử dụng để làm gì |
|---|---|---|
| File video | Upload MP4, AVI hoặc MOV; hoặc chọn video đã có trong danh sách | Đọc hình ảnh theo thời gian để phân tích |
| Camera | Nhập mã như `HAM_B1_CAM01` ở thanh bên trái | Gắn nhãn nguồn trong sổ cảnh báo và ảnh ghi nhận |
| Chế độ phân tích | Chọn trên tab Phân tích video | Quyết định phân tích từng người theo thời gian hay chấm điểm cả video |
| Nguồn đặc trưng và điểm đánh giá | Chọn khi bật trích xuất tư thế trong chế độ theo dõi | Quyết định dùng điểm baseline, MIL tư thế hoặc ResNet50 cho quy tắc đánh giá |

**Trường Camera chỉ là mã ghi chú nguồn.** Nhập địa chỉ IP hoặc tên camera vào đây không kết nối camera. Bản hiện tại chưa nhận URL RTSP, webcam hoặc link YouTube làm luồng đầu vào trực tiếp.

Video nên có FPS không đổi, gọi là **CFR**. Chế độ theo dõi yêu cầu FPS nguồn từ **4 trở lên**. Nên dùng video camera thật, thấy đủ người và xe, ít cắt cảnh. Video hình vẽ trong `data/demo` chỉ thử luồng chương trình; mô hình phát hiện có thể không nhận hình vẽ là người.

### 2.2 Cách A Upload video từ máy

1. Mở tab **Phân tích video**.
2. Tại **Tải video MP4 AVI MOV (CFR)**, bấm **Upload** và chọn file.
3. Chờ phần **Video nguồn** xuất hiện; có thể bấm Play để xem trước.
4. Chọn chế độ và các tùy chọn cần dùng, sau đó bấm **Phân tích**.

Giới hạn dung lượng xem tại nút Upload; cấu hình mặc định hiển thị 200 MB mỗi file. Nếu video vượt giới hạn, có thể dùng cách B với file MP4 có sẵn trong thư mục dự án.

**File đang upload được ưu tiên hơn video trong danh sách.** Muốn chuyển sang video khác, hãy bấm nút xóa file upload trước, rồi chọn lại video. Chọn tên khác trong danh sách khi vẫn còn file upload sẽ không đổi nguồn phân tích.

### 2.3 Cách B Chọn video đã có trong dự án

Chọn tên file ở **Hoặc video đã tải về**. Danh sách lấy các file `.mp4` trong:

```text
data/external_test/
data/demo/
data/videos/
```

Có thể chép thêm MP4 vào các thư mục này rồi tải lại trang để cập nhật danh sách. AVI và MOV dùng qua Upload.

Ví dụ video hiện có trên máy:

| Tên file | Mục đích thử |
|---|---|
| `tuoitre_ham_chung_cu_20230731.mp4` | Tương tác với xe trong khu đỗ xe, có cắt cảnh và phóng hình |
| `voh_orchard_parkview_20191008.mp4` | Khu vực barie ở tầng hầm |
| `youtube_a1bcB0_RMiQ.mp4` | Đối chứng bãi xe ban đêm, giả định hoạt động bình thường |
| `youtube_5BSCCBLD-_M.mp4` | Đối chứng bãi giữ xe, giả định hoạt động bình thường |

Video bên ngoài không được đóng gói trong Git hoặc ZIP ban đầu, nên máy khác có thể chưa có các tên này. Link nguồn và hướng dẫn demo nằm trong [HUONG_DAN_DEMO.md](HUONG_DAN_DEMO.md).

## 3 Chọn cách phân tích và mô hình sẽ làm gì

### 3.1 Theo dõi trực tiếp trên video

Dùng chế độ này khi muốn xem khung người, điểm theo thời gian, mức xanh–vàng–đỏ và ảnh ghi nhận tự động. Chữ **trực tiếp** ở đây có nghĩa là ứng dụng hiển thị từng khung đang xử lý từ file đã chọn.

Luồng xử lý:

```text
Video đã chọn
    → Đọc và lấy mẫu khung hình
    → Phát hiện người và xe máy
    → Gán ID theo dõi tạm thời
    → Trích xuất hình ảnh, tư thế và tương tác tùy cấu hình
    → Tính điểm bằng mô hình đã chọn
    → Áp dụng quy tắc theo thời gian để xác định mức đánh giá
    → Vẽ khung, hiển thị chỉ số và chụp ảnh nếu cần
    → Xuất video kết quả và nhật ký JSON
```

| Tùy chọn | Khi bật hoặc chọn |
|---|---|
| **Trích xuất tư thế và tương tác người–xe** | Thêm khung xương, chuyển động cổ tay, thời gian ở gần xe và các chỉ số tương tác. Mục tiêu hiển thị tối đa 4 FPS; đặc trưng hành vi cập nhật khoảng 2 lần/giây |
| **Baseline toàn cảnh** | Dùng điểm ExtraTrees của toàn cảnh làm nguồn điểm cho quy tắc; các bằng chứng tư thế vẫn được bổ sung nếu đã bật |
| **MIL tư thế (thử nghiệm)** | Dùng head học từ các đặc trưng tư thế và theo dõi người. MIL là cách học bằng nhãn cả video; chưa có nhãn đúng/sai cho từng người |
| **ResNet50 theo repo tham khảo (thử nghiệm)** | Trích xuất vector ảnh vùng người bằng ResNet50 và chấm điểm bằng head MIL do dự án huấn luyện. Tư thế/tương tác vẫn phục vụ giải thích và quy tắc màu |
| **Phát theo thời gian gốc** | Giới hạn tốc độ hiển thị để gần với thời gian video. Bỏ chọn để xử lý nhanh theo sức máy; máy chậm vẫn có thể xử lý chậm hơn video |
| **Tự chụp ảnh khi xuất hiện khung vàng hoặc đỏ** | Tự lưu ảnh và tạo cảnh báo chờ xác minh khi một người được đánh dấu cần xem xét |

Nếu tắt trích xuất tư thế, ứng dụng dùng baseline và mục tiêu theo dõi tối đa 8 FPS. Điều kiện lên đỏ cần các bằng chứng tư thế, nên cấu hình thiếu tư thế sẽ không đủ điều kiện đó.

Lần đầu dùng ResNet50 có thể cần tải khoảng **98 MB** trọng số chính thức; các lần sau dùng cache trong máy. Đây là tải mô hình, không phải huấn luyện lại.

Điểm **cảnh** chờ khoảng 4 giây đầu để tích lũy dữ liệu, sau đó cập nhật mỗi 4 giây với tối đa 48 giây quá khứ. Điểm từng người có thể trống khi chưa đủ quan sát hoặc chưa trích xuất được đặc trưng phù hợp. Vì vậy, điểm không nhất thiết thay đổi ở mọi khung hình.

### 3.2 Phân tích tổng quan

Dùng chế độ này để nhận một kết quả cấp video và kiểm tra chất lượng đầu vào. Chế độ này không vẽ khung theo từng người và không tự xác định người/frame cần chụp ảnh.

Mặc định hệ thống lấy tối đa 12 cửa sổ dài 4 giây trải đều video, trích xuất đặc trưng hình ảnh/chuyển động và chấm điểm video bằng baseline. Cách lấy mẫu có thể bỏ lỡ sự kiện ngắn nằm giữa các cửa sổ.

Bật **Phân tích liên tục các cửa sổ 4 giây (chậm hơn)** để duyệt các cửa sổ trên toàn bộ video và xem thêm đường điểm theo thời gian. **Điểm video** vẫn dùng cách biểu diễn 12 cửa sổ như lúc huấn luyện; đường điểm dùng ngữ cảnh trượt nên có thể khác điểm gộp cả video.

## 4 Ý nghĩa các chỉ số khi theo dõi

### 4.1 Các ô chỉ số lớn

| Chỉ số trên web | Ý nghĩa | Cách đọc đúng |
|---|---|---|
| **Thời gian video** | Vị trí hiện tại trong video nguồn, tính bằng giây | `18.0s` là giây thứ 18 trong clip, không phải đã chạy trên máy đúng 18 giây |
| **Điểm nghi vấn CẢNH** | Điểm baseline của toàn khung cảnh, thường trong khoảng 0–1 | Điểm cao hơn là tín hiệu đáng xem xét hơn theo baseline; không xác định người nào gây ra tín hiệu |
| **Nghi vấn / Nguy hiểm** | Số người khung vàng / khung đỏ trong khung đang hiển thị | `2 / 1` nghĩa là hiện có 2 người mức vàng và 1 người mức đỏ; không phải tổng sự kiện của cả video |
| **FPS xử lý thực đo** | Tốc độ xử lý và cập nhật của phiên theo dõi | Không phải FPS camera hoặc độ chính xác. Giá trị chịu ảnh hưởng sức máy, số người và tùy chọn phát theo thời gian gốc |

**Ngay cả khi chọn ResNet50, ô Điểm nghi vấn CẢNH vẫn là điểm baseline.** Mức đánh giá khi đó dùng điểm ở cột **ResNet50 thử nghiệm** cùng các bằng chứng tương tác. Hai điểm khác nhau là bình thường.

### 4.2 Bảng thông tin từng người

| Trường | Ý nghĩa |
|---|---|
| **ID** | Mã theo dõi tạm thời trong lượt xử lý. Không phải danh tính cư dân; cùng một người có thể đổi ID khi bị che hoặc khi video cắt cảnh |
| **Độ tin cậy người** hoặc `det` trên khung | Độ tin cậy rằng vùng khoanh là người. `0.95` không có nghĩa người đó có 95% khả năng ăn trộm |
| **Gần xe (giây)** | Thời gian ID được quan sát liên tục ở gần các xe máy phát hiện được. Có thể reset khi mất theo dõi; không xác minh quyền sở hữu hoặc quyền di chuyển xe |
| **Điểm CẢNH** hoặc `scene` | Điểm toàn cảnh, có thể giống nhau ở nhiều người trong cùng một khung |
| **MIL thử nghiệm** | Điểm từ head tư thế/hành vi V2. Có thể trống khi thiếu quan sát hoặc khớp đủ tin cậy |
| **ResNet50 thử nghiệm** hoặc `ResNet MIL` | Điểm từ head sử dụng đặc trưng ảnh vùng người. Có thể trống nếu chưa chọn nhánh ResNet50 hoặc chưa đủ mẫu |
| **Mức đánh giá** | An toàn, Nghi vấn, Nguy hiểm hoặc Chưa đủ dữ liệu theo chính sách hiện tại |
| **Lý do** | Các điều kiện quan sát khiến hệ thống giữ hoặc nâng mức; cần đọc cùng điểm và video |
| **Điều kiện còn thiếu / giới hạn** | Liệt kê điểm dưới ngưỡng, thiếu khớp, thời gian gần xe hoặc tín hiệu tay/đầu chưa đạt. Không phải bằng chứng người đó không có hành vi đáng ngờ |
| **Tỷ lệ khớp qua ngưỡng tin cậy** | Phần khớp thân/tay/chân được dùng trích xuất có confidence từ 0,4 trở lên. `0.75` nghĩa là khoảng 75% nhóm khớp đó qua ngưỡng, không phải mức chính xác nhận dạng hành vi |
| **Tay gần vùng xe** | Dấu hiệu hình học 2D về cổ tay gần vùng xe. Không chứng minh tay đang chạm ổ khóa hoặc phá khóa. Đọc cùng chất lượng tư thế; trống là chưa có dữ liệu |
| **Chuyển động tay** | Tốc độ cổ tay tương đối với thân người, đã chuẩn hóa kích thước. Không phải m/s và không phải xác suất bẻ khóa |
| **Cần xem xét** | `True` khi mức vàng hoặc đỏ. `False` có thể là xanh **hoặc chưa đủ dữ liệu**, nên phải đọc thêm cột Mức đánh giá |
| **Ngồi thấp/co gối (ước lượng)** | Tín hiệu hình học từ hai chân; trống nếu thiếu khớp chân tin cậy. Không phân biệt chắc chắn ngồi xổm với ngồi ghế/sửa xe |
| **Góc gối trung bình (độ)** | Trung bình hai góc hông–gối–cổ chân trên ảnh 2D; chỉ hiển thị khi đủ khớp. Không phải điểm nghi vấn |
| **Đủ dữ liệu đánh giá** | Có điểm, chất lượng, theo dõi và, khi bật tư thế, đủ quan sát tư thế tin cậy. `False` tương ứng trạng thái xám |

Các điểm cảnh/MIL/ResNet50 **chưa được hiệu chuẩn thành xác suất phạm tội**. Không diễn giải `0.80` là “80% chắc chắn trộm”, và không tự cộng/trung bình các điểm để suy ra kết luận.

### 4.3 Màu khung và điều kiện chuyển mức

| Màu và nhãn | Ý nghĩa sử dụng |
|---|---|
| **Xanh lá An toàn** — `SAFE` | Có đủ dữ liệu sơ bộ và chưa hình thành chuỗi dấu hiệu cần xem xét. Không bảo đảm không có sự kiện ở điểm mù hoặc bị mô hình bỏ sót |
| **Vàng Nghi vấn** — `SUSPICIOUS` | Có điểm cao khi ở gần xe đủ lâu, hoặc nhiều dấu hiệu tương tác kết hợp cần xem xét |
| **Đỏ Nguy hiểm** — `HIGH RISK` | Có điểm cao hơn cùng nhiều dấu hiệu cục bộ kéo dài: tay gần vùng xe, cúi người hoặc co gối/ngồi thấp, và chuyển động tay; cần ưu tiên xác minh |
| **Xám Chưa đủ dữ liệu** — `OBSERVING` | Thiếu điểm, theo dõi, chất lượng hoặc tư thế tin cậy khi bật phân tích tư thế. Nếu còn thiếu khớp chân/cổ tay và chưa có chuỗi tương tác cảnh báo thì cũng hiển thị xám. Đây là trạng thái chất lượng, không phải lớp hành vi thứ tư |

Để giảm nhấp nháy, tín hiệu vàng cần giữ khoảng 1 giây, đỏ khoảng 2 giây; hạ mức sau khoảng 2 giây. Đỏ còn cần các điều kiện gần xe/tay gần vùng xe và tư thế đủ tin cậy. **Điểm cao đơn lẻ không đủ lên đỏ.**

Ngưỡng tùy nguồn điểm: ví dụ hiện tại baseline có ngưỡng vàng 0,50; nhánh ResNet50 có ngưỡng vàng 0,41. Ngưỡng đỏ hiện thường là 0,80, kèm điều kiện hành vi. Giá trị của lượt chạy xem ở phần kết quả và `risk_policy` trong JSON; các ngưỡng màu là quy tắc thử nghiệm, chưa có kiểm định ba mức độc lập.

Đeo khẩu trang hoặc nhìn quanh riêng lẻ không làm tăng mức. Hệ thống chưa có bộ nhận dạng được huấn luyện riêng để xác nhận hành động bẻ khóa. Vùng xe và tư thế 2D có thể bị nhầm do che khuất, góc camera hoặc hành động lấy xe hợp lệ.

## 5 Ý nghĩa kết quả Phân tích tổng quan

| Chỉ số hoặc kết luận | Ý nghĩa |
|---|---|
| **Điểm video** | Điểm gộp cả video từ baseline; không phải điểm riêng một người |
| **Ngưỡng từ validation** | Mốc chọn bằng tập validation để so sánh với điểm video. Khác với toàn bộ điều kiện lên vàng/đỏ của chế độ theo dõi |
| **Thời gian xử lý** | Thời gian suy luận được ghi nhận. Có thể khác tổng thời gian chờ trên web vì còn chuẩn bị video, tải/nạp mô hình hoặc hiển thị |
| **CẦN BẢO VỆ XEM XÉT** — `review` | Điểm video đạt ngưỡng và đủ chất lượng sơ bộ; cần người xem lại |
| **ĐIỂM NGHI VẤN THẤP** — `low_score` | Điểm dưới ngưỡng; có thể vẫn có sự kiện bị bỏ sót |
| **KHÔNG ĐỦ CHẤT LƯỢNG** — `insufficient_quality` | Không đủ số cửa sổ vượt kiểm tra chất lượng sơ bộ để đưa ra đánh giá đáng tin |

Biểu đồ có trục ngang là **giây trong video**, trục dọc là **điểm ngữ cảnh**. Đường tăng cho thấy mô hình chấm ngữ cảnh đó cao hơn; đỉnh đường chưa được xác minh là thời điểm xảy ra trộm.

Bảng cửa sổ có các trường sau:

| Trường | Ý nghĩa |
|---|---|
| `start`, `end` | Thời điểm đầu/cuối cửa sổ, đơn vị giây |
| `score` | Điểm của ngữ cảnh tương ứng; chế độ lấy mẫu có thể lặp lại điểm video |
| `brightness` | Độ sáng trung bình trên vùng nội dung, theo thang ảnh xám 0–255 |
| `blur` | Chỉ số phương sai Laplacian; số thấp có thể liên quan ảnh mờ hoặc cảnh ít kết cấu, không phải điểm phần trăm |
| `dark_fraction`, `saturated_fraction` | Tỷ lệ pixel quá tối/quá sáng, khoảng 0–1 |
| `valid_frames` | Số khung đọc được để trích xuất trong cửa sổ, không phải tổng frame của video nguồn |
| `usable` | Cửa sổ vượt các kiểm tra chất lượng sơ bộ hay không |
| `low_source_fps` | FPS nguồn dưới mức lấy mẫu cần dùng |

Các kiểm tra sáng/mờ hiện dùng ngưỡng minh họa, chưa được hiệu chỉnh riêng cho từng camera.

## 6 Output nhận được và cách lấy ra

### 6.1 Kết quả của chế độ theo dõi

| Output | Xem hoặc tải ở đâu | Nội dung |
|---|---|---|
| Khung đang xử lý | Vùng **Đang theo dõi và chấm điểm** | Hình ảnh có khung người, ID, điểm, màu; có khung xương nếu bật tư thế |
| Video kết quả | **Phát lại video có khung và điểm theo thời gian** → **Tải video có khung** | MP4 H.264 có đánh dấu, không có âm thanh |
| Nhật ký JSON | **Tải nhật ký theo dõi JSON** | Nguồn video, cấu hình, điểm, mức/lý do theo từng người và thời điểm, thông tin ảnh chụp |
| Ảnh ghi nhận | **Ảnh ghi nhận tự động** hoặc **Sổ cảnh báo** | Ảnh nguồn PNG, ảnh đánh dấu JPG và metadata JSON |

Video kết quả và JSON tổng được lưu trong `runtime/annotated/` khi chạy hoàn tất. Video được lấy mẫu và thu nhỏ phục vụ hiển thị, nên số frame/độ phân giải có thể khác nguồn. **PNG ghi nhận giữ độ phân giải frame nguồn đã giải mã.**

Khi bấm **Dừng xử lý**, video đầu ra đầy đủ chưa được tạo; ảnh đã lưu trước đó vẫn còn trong Sổ cảnh báo. Chạy lại một video tạo lượt phân tích mới, không tiếp tục từ điểm đã dừng.

### 6.2 Ảnh ghi nhận tự động

Giữ bật **Tự chụp ảnh khi xuất hiện khung vàng hoặc đỏ**. Mỗi lần ghi nhận tạo thư mục:

```text
runtime/evidence/<run_id>/<evidence_id>/
├── original.png
├── annotated.jpg
└── metadata.json
```

| Tệp | Nội dung |
|---|---|
| `original.png` | Đúng frame nguồn ở thời điểm hệ thống đánh dấu, chưa vẽ lên ảnh; không lấy frame đầu video thay thế |
| `annotated.jpg` | Ảnh tại cùng thời điểm có khung và điểm để người xem kiểm tra nhanh |
| `metadata.json` | Camera, nguồn/hash video, thời gian, chỉ số frame, ID, tọa độ, mức, lý do, điểm, ngưỡng và hash ảnh |

Các khung liên tiếp của cùng ID được gộp thành một đợt. Khoảng ngắt tối đa 3 giây vẫn được gộp; hai lần chụp cùng mức của cùng ID cách nhau tối thiểu 10 giây. Khi nâng **vàng → đỏ**, hệ thống chụp thêm ngay để giữ thời điểm tăng mức.

Ví dụ minh họa một phần metadata, không phải kết quả của một vụ việc đã xác minh:

```json
{
  "camera_id": "HAM_B1_CAM01",
  "video_time_seconds": 18.0,
  "source_frame_index": 540,
  "source_frame_index_base": 0,
  "source_fps": 30.0,
  "track_ids": [7],
  "risk_level_at_capture": "suspicious",
  "alarm_score": 0.62,
  "threshold": 0.5,
  "status_at_capture": "pending"
}
```

Đọc ví dụ: tại giây 18 của video 30 FPS, ID 7 được đánh dấu nghi vấn; hệ thống lưu frame nguồn số 540 theo cách đếm từ 0. Điểm 0,62 chưa phải xác suất phạm tội. Các điều kiện chi tiết nằm ở thông tin từng người trong JSON đầy đủ.

`captured_at_utc` là thời điểm ứng dụng lưu ghi nhận theo UTC, không phải giờ camera đã quay sự kiện. `peak_alarm_score` là điểm cao nhất quan sát trong đợt, có thể xuất hiện sau ảnh đầu đợt; `alarm_score` là điểm lúc chụp. Hash hỗ trợ kiểm tra tệp có thay đổi, không xác nhận kết luận của mô hình.

### 6.3 Kết quả của Phân tích tổng quan

Bấm **Tải kết quả JSON** để nhận điểm video, quyết định, chất lượng và các cửa sổ được phân tích. Nút **Lưu cảnh báo chờ xác minh** lưu kết quả cấp video; chế độ này không tự chỉ ra người hoặc frame cần chụp. Muốn có ảnh theo người và thời điểm, dùng chế độ theo dõi.

## 7 Xem và xử lý Sổ cảnh báo

1. Mở tab **Sổ cảnh báo**.
2. Chọn **Mã cảnh báo** cần xem.
3. Đọc mức đánh giá, thời gian video và ID; mở ảnh đánh dấu để đối chiếu.
4. Tải ảnh gốc hoặc metadata khi cần kiểm tra kỹ hơn.
5. Chọn **Kết luận của bảo vệ**, nhập ghi chú và bấm **Cập nhật kết luận**.

| Trạng thái | Ý nghĩa |
|---|---|
| `pending` | Chờ người xác minh; trạng thái mặc định của cảnh báo tự tạo |
| `confirmed` | Người vận hành đã xác nhận sự kiện cần xử lý theo quá trình kiểm tra |
| `false_alarm` | Người vận hành đánh giá là báo nhầm, ví dụ chủ xe hoặc nhân viên di chuyển xe hợp lệ |

Mức xanh/vàng/đỏ là đánh giá của hệ thống; trạng thái trên là kết luận xử lý của người vận hành. Hai trường có ý nghĩa khác nhau. Lịch sử cập nhật được lưu và vai trò Quản trị có thể xem nhật ký. Đổi trạng thái không tự huấn luyện lại mô hình hoặc xóa ảnh.

Các dữ liệu này lưu cục bộ trong `runtime/events.sqlite3` và thư mục ảnh. Không có chức năng gửi SMS, Telegram hoặc thông báo đẩy trong bản hiện tại.

## 8 Đọc tab Kết quả mô hình

Tab này hiển thị số đo từ thí nghiệm dữ liệu đã gán nhãn, **không phải độ chính xác vừa đo trên video bạn upload**. Với video mới chưa có nhãn, hệ thống không tự biết dự đoán nào đúng hay sai để tính precision/recall.

| Chỉ số | Ý nghĩa |
|---|---|
| **Precision** | Trong các video được dự đoán thuộc lớp Stealing, tỷ lệ khớp nhãn thật. Thấp nghĩa là có nhiều báo nhầm |
| **Recall** | Trong các video có nhãn Stealing, tỷ lệ mô hình phát hiện được. Thấp nghĩa là bỏ sót nhiều |
| **F1** | Cân bằng precision và recall. Giá trị cao cần cả hai cùng đủ tốt |
| **Accuracy** | Tỷ lệ video được phân loại đúng trên toàn tập đánh giá; phải đọc cùng phân bố lớp và số bỏ sót |
| **ROC-AUC / AUC** | Khả năng xếp hạng video dương cao hơn video âm khi xét nhiều ngưỡng; không phải xác suất trộm của một người |
| **AP / Average precision** | Tóm tắt quan hệ precision–recall theo thứ tự điểm; chịu ảnh hưởng tỷ lệ lớp và tập đánh giá |
| **F1 validation** | Kết quả trên tập dùng chọn cấu hình/ngưỡng. Không phải kết quả kiểm thử độc lập cuối cùng |
| **F1 test cũ / AUC test cũ** | Kết quả chẩn đoán trên tập test đã được xem trong quá trình phát triển. Không đại diện độ chính xác thực tế ở hầm |
| **Chọn bằng validation** | Cấu hình được chọn theo tiêu chí validation, không nhất thiết là dòng có điểm test cao nhất |

Với confusion matrix, hàng là nhãn thật và cột là dự đoán, theo thứ tự Normal rồi Stealing:

| | Dự đoán Normal | Dự đoán Stealing |
|---|---|---|
| **Thật Normal** | TN — bình thường được nhận đúng | FP — báo nhầm |
| **Thật Stealing** | FN — bỏ sót | TP — phát hiện đúng |

Các số hiện tại đánh giá chủ yếu **cấp video**, chưa đo độ chính xác ba mức theo từng người hoặc khả năng nhận riêng động tác bẻ khóa. Số khung có cảnh báo cũng không phải số vụ trộm hay số dự đoán đúng.

## 9 Quy trình chạy thử gợi ý

1. Chọn vai trò Bảo vệ và nhập mã Camera để dễ tra cứu.
2. Xóa file upload cũ nếu có; chọn một video thật cần kiểm tra.
3. Xem Video nguồn để chắc chắn đúng file và phát được.
4. Chọn Theo dõi trực tiếp trên video; bật trích xuất tư thế.
5. Chọn nguồn điểm muốn thử. Giữ Baseline khi cần đối chiếu; chọn ResNet50 để thử nhánh học sâu mới.
6. Bỏ Phát theo thời gian gốc nếu muốn xử lý nhanh; giữ bật tự chụp ảnh.
7. Bấm Phân tích. Đọc màu, mức và lý do cùng video; không chỉ nhìn một điểm số.
8. Khi xong, tải video/JSON và kiểm tra ảnh trong Sổ cảnh báo.
9. Thử cả video hoạt động hợp lệ để ghi nhận báo nhầm, thay vì chỉ trình diễn ví dụ mô hình đoán đúng.

Nếu đổi mô hình hoặc tùy chọn, bấm Phân tích để có lượt kết quả mới. Video kết quả đang hiển thị có thể là lượt hoàn tất trước đó; đọc thêm nguồn cảnh báo trong phần kết quả.

## 10 Giải đáp khi kết quả khó hiểu

| Hiện tượng | Giải thích và thao tác |
|---|---|
| Chọn video mới nhưng vẫn xử lý file cũ | Xóa file trong Upload trước; file upload đang được ưu tiên |
| Video phát được nhưng chưa có điểm | Điểm cảnh cần tích lũy 4 giây; điểm từng người cần đủ quan sát. Đợi và kiểm tra cột Chưa đủ dữ liệu |
| `det` cao nhưng khung xanh hoặc xám | Mô hình tin rằng có người, chưa có nghĩa có hành vi đáng ngờ hoặc đủ dữ liệu đánh giá |
| Điểm cảnh cao nhưng không có khung đỏ | Có thể đang dùng điểm MIL/ResNet50 cho màu, hoặc chưa đủ điều kiện hành vi/thời gian |
| Không có ảnh tự chụp | Kiểm tra đã bật tùy chọn chụp và đã có mức vàng/đỏ. Một đợt đã chụp sẽ được gộp, không chụp lại mọi frame |
| Cùng người đổi ID | Có thể do che khuất, kích thước nhỏ, cắt cảnh hoặc lỗi tracking; ID không phải nhận diện danh tính |
| Hình vẽ không có khung người | Detector được huấn luyện trên ảnh thật; đổi sang video camera thật |
| ResNet50 chờ lâu ở lần đầu | Đang tải/nạp encoder; cần mạng cho lần tải đầu, sau đó dùng cache |
| `unexpected keyword argument` sau cập nhật code | Dừng Streamlit bằng Ctrl+C rồi chạy lại từ `.venv`; xem thêm hướng dẫn demo kỹ thuật |
| Màu đỏ xuất hiện với chủ xe lấy xe | Có thể là báo nhầm. Xác minh và ghi `false_alarm`; không đổi ngưỡng chỉ để một video cho kết quả mong muốn |

Tài liệu liên quan: [hướng dẫn demo và xử lý lỗi](HUONG_DAN_DEMO.md), [thử nghiệm đặc trưng hành vi](NANG_CAP_HANH_VI.md), [tích hợp ResNet50 và quy tắc ba mức](TICH_HOP_DILANIU.md).
