# Cảnh báo nghi vấn lấy trộm xe máy trong hầm chung cư

**[Hướng dẫn sử dụng hệ thống](reports/HUONG_DAN_SU_DUNG_HE_THONG.md)** — cách đưa video vào, mô hình xử lý, output và ý nghĩa các chỉ số trên web.

Bộ bài tập gồm mã nguồn Python, mô hình đã huấn luyện trên video thật thuộc UCF-Crime, giao diện Streamlit, báo cáo tiếng Việt và kết quả chạy. Mô hình nền hiện phân loại **Stealing / Normal ở cấp video**. Chưa có bằng chứng mô hình nhận dạng chính xác trộm **xe máy tại hầm chung cư**; báo cáo phân biệt rõ phần đã thực nghiệm và thiết kế mở rộng cần dữ liệu tại hầm.

## Chạy nhanh

Python 3.12–3.13, máy có CPU; bản đã kiểm thử dùng Python 3.13 trên macOS. Không cần GPU, API key hoặc Kaggle token. Phiên bản scikit-learn được cố định để đọc đúng mô hình đã lưu; `requirements-lock.txt` ghi toàn bộ môi trường thực nghiệm.

```bash
git clone https://github.com/mynameisFuong/practice_XLA.git
cd practice_XLA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/setup_detector.py
python scripts/make_demo.py
python -m streamlit run app.py
```

Windows PowerShell: dùng `.venv\Scripts\Activate.ps1` thay lệnh `source`. Mở `http://localhost:8501`. Chọn video thật hoặc tải video MP4 lên. Video nguồn được chuyển sang H.264 và xem được ngay trước khi phân tích. Chọn chế độ **Theo dõi trực tiếp trên video** để xem khung người và điểm cập nhật; chọn **Phân tích tổng quan** để dùng chế độ đánh giá video ban đầu.

## Xem video với khung người và điểm cập nhật

Ứng dụng **tự chụp ảnh khi xuất hiện khung vàng hoặc đỏ** (mặc định bật). Mỗi đợt nghi vấn lưu frame nguồn PNG, ảnh có khung JPG và JSON mức/thời gian/ID/điểm/hash tại `runtime/evidence`; đồng thời tạo cảnh báo pending. Gộp theo ID và giới hạn 10 giây giữa các ảnh, riêng chuyển vàng → đỏ chụp thêm ngay. Ảnh đã lưu vẫn còn nếu dừng xử lý. Xem [hướng dẫn chụp ảnh](reports/HUONG_DAN_DEMO.md).

Đã bổ sung detector YOLOv8n pretrain trên COCO và ByteTrack để phát hiện/theo dõi người và xe máy. Mô hình này tạo khung đối tượng, không được huấn luyện nhận dạng trộm. Trọng số lưu tại `artifacts/yolov8n.pt`; nguồn và hash tại `artifacts/detector_source.json`.

- Khung xanh: đủ dữ liệu sơ bộ nhưng chưa có chuỗi dấu hiệu; vàng: nghi vấn; đỏ: nguy cơ cao cần xác minh. Xám là trạng thái chưa đủ dữ liệu, không được coi là an toàn.
- `scene`: điểm nghi vấn của **cảnh**, từ ExtraTrees hiện có; không phải xác suất ăn trộm của người trong khung.
- Khung đỏ `HIGH RISK`: cần đồng thời điểm cao và chuỗi dấu hiệu cục bộ đủ thời gian: tay gần vùng xe, cúi người, chuyển động tay và gần xe. Khẩu trang hay nhìn quanh riêng lẻ không tăng mức. Chưa có bộ phân lớp bẻ khóa được huấn luyện; đây là chính sách thử nghiệm trên ảnh 2D.
- Bốn giây đầu tích lũy dữ liệu; điểm cảnh cập nhật mỗi 4 giây bằng tối đa 48 giây quá khứ, không lấy frame tương lai. Khung người cập nhật ở FPS xử lý mục tiêu 8. Đây là phát luồng từ file, chưa phải kết nối RTSP; FPS thực đo hiển thị trên web.
- Khi hoàn tất có video H.264 với khung và điểm để phát lại/tải xuống, cùng JSON theo dõi. Video kết quả không có âm thanh.

Video tổng hợp hình vẽ có thể không được detector COCO nhận là người. Nên dùng video thật trong `data/external_test`. Chất lượng thấp, người quá nhỏ, che khuất hoặc chuyển góc có thể làm mất khung/đổi ID; ID không xác định danh tính thật. Chưa có đánh giá precision/recall cho quy tắc khung đỏ, và việc thêm detector không làm thay đổi các số đo mô hình phân loại bên dưới.

```bash
python scripts/annotate_video.py data/external_test/tuoitre_ham_chung_cu_20230731.mp4 --out runtime/annotated/tuoitre.mp4
```

Nguồn kỹ thuật: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), [tracking](https://docs.ultralytics.com/modes/track/), [Streamlit video và vấn đề codec MP4V](https://docs.streamlit.io/develop/api-reference/media/st.video). FFmpeg đi kèm thư viện `imageio-ffmpeg`, không cần cài FFmpeg hệ thống để phát video trong app.

## Thử nhánh ResNet50 từ repo tham khảo

Đã kiểm tra repo Dilaniu và bổ sung encoder ResNet50 với head MIL do dự án này tự huấn luyện. Trong chế độ theo dõi, bật tư thế và chọn **ResNet50 theo repo tham khảo (thử nghiệm)** tại **Nguồn đặc trưng và điểm đánh giá**. Trọng số ImageNet chính thức tải khoảng 98 MB vào runtime/models lần đầu, không đóng gói trong Git hoặc ZIP. [Báo cáo tích hợp](reports/TICH_HOP_DILANIU.md) ghi audit repo, kết quả và chính sách ba màu.

Nhánh được chọn bằng validation có F1 validation 0,80, AUC test cũ 0,75 và F1 test cũ 0,545. Các số test chỉ là chẩn đoán trên tập đã xem, không phải độ chính xác thực tế. Hai video đối chứng bình thường giả định vẫn vượt ngưỡng MIL; giữ baseline làm mặc định và dùng nhánh mới để thử nghiệm. Repo tham khảo không cung cấp checkpoint trộm riêng và dự án này không sử dụng quy tắc khoảng cách khuôn mặt của notebook.

```bash
python scripts/extract_visual.py
python scripts/train_behavior.py --visual
python scripts/annotate_video.py data/external_test/tuoitre_ham_chung_cu_20230731.mp4 --behavior --backend resnet_mil --no-evidence --out runtime/annotated/resnet_demo.mp4
```

Hai video trong `data/demo` là hình vẽ sinh bằng code, chỉ kiểm tra luồng chương trình, **không dùng huấn luyện và không chứng minh độ chính xác**. Video thật không nằm trong file nén; tải bằng script bên dưới hoặc dùng video nhóm có quyền sử dụng.

## Tái lập thí nghiệm

```bash
python scripts/download_data.py
python scripts/train.py
python scripts/predict.py data/videos/Stealing019_x264.mp4 --dense --out artifacts/prediction.json
python -m pytest -q
```

Script tải có giới hạn dung lượng từng video và giữ phân chia train/test theo metadata của bản sao dataset. Validation lấy từ train. `data/manifest.csv` lưu nhãn, split, URL, SHA-256 và tên video. Không trộn frame của cùng video giữa các tập. Bản sao cố định tại revision `8d783bb`. Số lượng thực tế và dung lượng nằm trong báo cáo và manifest. Mạng chậm có thể mất nhiều phút. Có thể chạy lại để tận dụng file đã tải. Nếu một URL hỏng, script báo lỗi và không bịa dữ liệu thay thế.

Các file `data/features/*.npz` đi kèm cho phép chạy lại `train.py` mà không tải video; chúng là đặc trưng rút ra từ video thật. Muốn kiểm chứng việc trích xuất, tải video rồi xóa thư mục cache features và huấn luyện lại. Mỗi cache gắn phiên bản đặc trưng và SHA-256 video.

Đầu ra `artifacts/metrics.json`, `test_predictions.csv`, `evaluation.png`, `model.joblib` do chạy thật tạo ra. Chọn mô hình và ngưỡng bằng validation; test chỉ báo cáo sau lựa chọn. Mô hình dự đoán nhãn video; đường điểm theo thời gian là thử nghiệm trên ngữ cảnh trượt, chưa đánh giá định vị.

## Huấn luyện với video hầm do nhóm thu thập

Chuẩn bị CSV có `path,label,split,session_id,camera_id`. Nhãn 0 = bình thường, 1 = có hành vi nghi vấn được người gán nhãn kiểm tra. Chia camera và buổi quay trước khi tạo clip. Script kiểm tra camera/session không xuất hiện ở nhiều split.

```bash
python scripts/import_local.py du_lieu_ham.csv --out data/local_manifest.csv
python scripts/train.py --manifest data/local_manifest.csv
```

Lệnh train ghi lại mô hình tại `artifacts/model.joblib`; sao lưu kết quả UCF trước nếu muốn giữ cả hai. Mỗi split phải có đủ hai nhãn. Không dùng lại test để chỉnh ngưỡng. Bản demo không có đăng nhập, RTSP liên tục, nhận diện danh tính/chủ sở hữu xe, nhận diện biển số, hoặc thông báo đẩy. Vai trò trong sidebar chỉ mô phỏng giao diện. Báo cáo PDF mô tả phiên bản baseline ban đầu; phần theo dõi vừa bổ sung được mô tả tại README và hướng dẫn demo.

## Kết quả hiện tại

Đã bổ sung tư thế và tương tác theo từng người: 67 giá trị mỗi thời điểm, gộp 201 giá trị mỗi cửa sổ. Có head MIL giám sát yếu và so sánh pose-only/interaction-only/kết hợp; head mới **chưa vượt baseline trên validation**, nên không thay mô hình mặc định. Hai video đối chứng YouTube bạn gửi đã được giữ ngoài train và cũng bộc lộ báo nhầm của MIL. Xem [báo cáo nâng cấp hành vi](reports/NANG_CAP_HANH_VI.md) để biết số đo, độ bao phủ, các giới hạn và cách chạy. Bật checkbox tư thế để xem khung xương; checkbox dùng MIL cho cảnh báo chỉ là chế độ thử nghiệm.

```bash
python scripts/extract_behavior.py
python scripts/train_behavior.py
python scripts/summarize_behavior.py
```

63 video thật: train 40, validation 10, test 13. ExtraTrees được chọn bằng validation; test ROC-AUC 0,65, recall 0,60, precision 0,429 và F1 0,50. Phát hiện 3/5 video Stealing, báo nhầm 4/8 video Normal. F1 còn thấp hơn Dummy dự đoán tất cả là dương. Đây là baseline học thuật còn yếu; cần dữ liệu đúng miền và nâng cấp mô hình trước khi xem là giải pháp thực tế.

## Tiền xử lý và video lỗi

CPU baseline: lấy tối đa 12 cửa sổ 4 giây trải đều video → lấy mẫu 4 FPS → giữ tỷ lệ và letterbox 128×96 → Gaussian nhẹ → CLAHE trên vùng nội dung → histogram gradient, HSV, LBP và optical flow → thống kê → bộ phân lớp. Chế độ dense duyệt hết cửa sổ không chồng lấn, thời gian xử lý tăng theo độ dài video.

OpenCV trong demo giả định video có tốc độ khung hình không đổi (CFR). Với VFR hoặc thời gian nguồn không ổn định, chuyển trước bằng FFmpeg nếu máy có cài:

```bash
ffmpeg -i input.mp4 -vf fps=25 -c:v libx264 -crf 20 -an output_cfr.mp4
```

Chuyển CFR không khôi phục thông tin đã mất. FPS nguồn thấp, thiếu frame, quá tối hoặc mờ sẽ xuất hiện trong bảng chất lượng. Kiểm tra chất lượng là heuristic minh họa, chưa hiệu chuẩn cho camera thật. Không thể biến ảnh mất chi tiết thành bằng chứng đáng tin chỉ bằng tăng sáng hoặc phóng to.

## Tệp nộp bài

- `reports/BAO_CAO.pdf`: báo cáo để nộp; `reports/BAO_CAO.html` là bản mở trong trình duyệt.
- `reports/BAO_CAO.md`: nội dung dễ sửa.
- `reports/DANH_SACH_NHOM.csv`: điền tên, MSSV, điểm cộng, tỷ lệ đóng góp thực tế; tổng đóng góp bằng 100%.
- `artifacts`: mô hình và kết quả chạy thật.
- `src`, `scripts`, `app.py`, `tests`: source code và kiểm thử.
- `data/manifest.csv`, metadata, đặc trưng: nguồn gốc và dữ liệu tái lập nhỏ gọn.

Không điền sẵn thông tin thành viên hoặc tự nhận đã quay/kiểm chứng video hầm. Dataset gốc và bản sao không được khẳng định có giấy phép phân phối mở; bản nén không đóng gói các video thật. Nguồn nghiên cứu và đường tải được giữ trong báo cáo.
