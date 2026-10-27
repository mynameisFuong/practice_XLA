"""Local Streamlit demo: upload, infer, inspect, acknowledge and audit."""
from pathlib import Path
import hashlib,json,time,importlib
import pandas as pd
import streamlit as st
from src.predict import predict_video
import src.storage as storage_module
if getattr(storage_module,'STORAGE_API_VERSION',0)<2:
    importlib.reload(storage_module)
from src.storage import connect,create_event,update_event,list_evidence
from src.media import browser_video,source_key
from src.risk_levels import LABELS
import src.runtime_loader as runtime_loader

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Giám sát bãi xe',page_icon='🛵',layout='wide')
st.title('Hệ thống cảnh báo nghi vấn lấy trộm xe máy')
st.caption('Bản nghiên cứu trên UCF-Crime • Chạy CPU • Bảo vệ xác minh mọi cảnh báo')
st.warning('Mô hình hiện phân biệt video Stealing và Normal nói chung. Chưa được kiểm định riêng tại hầm chung cư; điểm số không phải xác suất phạm tội.')
role=st.sidebar.selectbox('Vai trò mô phỏng',['Bảo vệ','Quản trị','Cư dân'])
st.sidebar.caption('Chuyển vai trò để minh họa giao diện. Demo chưa có đăng nhập hoặc phân quyền bảo mật, chỉ chạy máy cá nhân.')
camera=st.sidebar.text_input('Camera','HAM_B1_CAM01')
tabs=st.tabs(['Phân tích video','Sổ cảnh báo','Kết quả mô hình'])
with tabs[0]:
    if role=='Cư dân':
        st.info('Thiết kế vận hành: cư dân chỉ nhận thông báo xe đã đăng ký qua hệ thống xác thực. Tính năng cư dân chưa triển khai trong bản demo.')
    else:
        available=(sorted((ROOT/'data/external_test').glob('*.mp4'))
                   +sorted((ROOT/'data/demo').glob('*.mp4'))
                   +sorted((ROOT/'data/videos').glob('*.mp4')))
        uploaded=st.file_uploader('Tải video MP4 AVI MOV (CFR)',type=['mp4','avi','mov'])
        chosen=st.selectbox('Hoặc video đã tải về',[None]+available,format_func=lambda p:'Chọn video' if p is None else p.name)
        path=chosen
        if uploaded is not None:
            folder=ROOT/'runtime/uploads';folder.mkdir(parents=True,exist_ok=True)
            content=uploaded.getvalue()
            path=folder/(hashlib.sha256(content).hexdigest()[:16]+Path(uploaded.name).suffix.lower())
            if not path.exists():path.write_bytes(content)
        if path:
            token=source_key(path)
            if st.session_state.get('input_token')!=token:
                for key in ['result','live_result','saved_event','captured_evidence']:
                    st.session_state.pop(key,None)
                st.session_state['input_token']=token
            st.subheader('Video nguồn')
            try:
                with st.spinner('Chuẩn bị video H.264 để phát trên trình duyệt...'):
                    playable=browser_video(path)
                st.video(str(playable))
            except Exception as e:st.error(str(e))
        mode=st.radio('Chế độ phân tích',['Theo dõi trực tiếp trên video','Phân tích tổng quan'],horizontal=True)
        is_live=mode=='Theo dõi trực tiếp trên video'
        if is_live:
            detector_choices={'YOLOv8n hiện tại':ROOT/'artifacts/yolov8n.pt'}
            if (ROOT/'model/yolo_behavior_model.pt').exists():
                detector_choices['Model bổ sung: yolo_behavior_model.pt (YOLO11 COCO)']=ROOT/'model/yolo_behavior_model.pt'
            detector_weights=detector_choices[st.selectbox('Model phát hiện người và xe',list(detector_choices))]
            profiles={'Nhanh (640 pixel)':'fast','Người nhỏ / bị che: toàn ảnh 1280 + 4 vùng phóng lớn':'small_people'}
            detector_profile=profiles[st.selectbox('Cách chạy detector',list(profiles))]
            st.caption('Video cổng hầm 424×240: nên thử model bổ sung cùng chế độ Người nhỏ. Chạy chậm hơn, có thể tăng phát hiện nhầm. Ứng viên yếu hoặc chưa có ID chỉ được quan sát, chưa đủ đánh giá hành vi.')
            st.caption('Detector tạo khung người/xe và ID theo dõi. Model bổ sung hiện có 80 lớp COCO, chưa có lớp trộm/bẻ khóa; độ tin cậy phát hiện không phải điểm hành vi. Chọn cùng nhánh chấm điểm bên dưới để so sánh hai detector.')
            filter_choices={'Giữ ảnh gốc':'off','Tự động khi tối hoặc tương phản thấp':'auto','Tăng tương phản cho video thiếu sáng':'low_light'}
            preprocessing=filter_choices[st.selectbox('Tiền xử lý cho phát hiện và tư thế',list(filter_choices))]
            denoise=st.checkbox('Lọc nhiễu nhẹ (bilateral)',value=False)
            sharpen=st.checkbox('Làm nét nhẹ',value=False)
            st.caption('Bộ lọc chỉ áp dụng cho detector và tư thế. Điểm cảnh/ResNet, chất lượng và ảnh gốc lưu bằng chứng vẫn dùng ảnh nguồn. Lọc không khôi phục được chi tiết đã mất; nên chạy đối chiếu ảnh gốc.')
            rich_behavior=st.checkbox('Trích xuất tư thế và tương tác người–xe',value=True)
            mil_alarm=False;backend='scene'
            if rich_behavior:
                st.caption('Bổ sung 17 điểm tư thế, co gối/ngồi thấp ước lượng, động tác cổ tay, thân người, thời gian tương tác và chuyển động người–xe. Thiếu tư thế tin cậy sẽ hiển thị xám thay vì xanh. Khung hình hiển thị tối đa 4 FPS, đặc trưng hành vi cập nhật 2 lần/giây.')
                choices={'Baseline toàn cảnh':'scene'}
                if (ROOT/'artifacts/behavior_v2/model.pt').exists():choices['MIL tư thế (thử nghiệm)']='pose_mil'
                if (ROOT/'artifacts/resnet_v3/model.pt').exists():choices['ResNet50 theo repo tham khảo (thử nghiệm)']='resnet_mil'
                backend=choices[st.selectbox('Nguồn đặc trưng và điểm đánh giá',list(choices))]
                mil_alarm=backend=='pose_mil'
                st.caption('Head MIL học bằng nhãn cả video. ResNet50 là encoder ImageNet; phần phân lớp do dự án này huấn luyện, không phải trọng số nhận dạng trộm được repo tham khảo cung cấp. Lần đầu cần tải khoảng 98 MB trọng số chính thức.')
            st.info('🟢 An toàn: chưa ghi nhận chuỗi dấu hiệu. 🟡 Nghi vấn: điểm và tương tác cần xem xét. 🔴 Nguy hiểm: nhiều dấu hiệu tác động xe kéo dài kèm điểm cao, cần xác minh. Xám: chưa đủ dữ liệu.')
            st.caption('Khẩu trang và nhìn quanh riêng lẻ không làm tăng mức. Bản hiện tại chưa có bộ nhận dạng được huấn luyện riêng cho bẻ khóa; màu đỏ là nguy cơ cao theo quy tắc thử nghiệm.')
            st.caption('Chưa có model phát hiện khẩu trang/mũ chuyên dụng. Tư thế ngồi thấp cũng có thể là chủ xe kiểm tra hoặc sửa xe; chỉ phối hợp cảnh báo khi có thêm tương tác tay–xe kéo dài.')
            st.caption('Chấm điểm cảnh chờ đủ 4 giây đầu và chỉ dùng quá khứ. Video hình vẽ có thể không được detector nhận là người; nên thử video thật.')
            pace=st.checkbox('Phát theo thời gian gốc',value=True)
            auto_capture=st.checkbox('Tự chụp ảnh khi xuất hiện khung vàng hoặc đỏ',value=True)
            st.caption('Lưu ảnh gốc, ảnh đánh dấu và mức đánh giá vào Sổ cảnh báo. Gộp theo ID; khi chuyển vàng → đỏ sẽ chụp thêm ngay.')
            dense=True
        else:
            dense=st.checkbox('Phân tích liên tục các cửa sổ 4 giây (chậm hơn)',value=False)
        if st.button('Phân tích',type='primary'):
            if not path:
                st.error('Hãy chọn một video.')
            elif is_live:
                st.session_state.pop('live_result',None)
                st.session_state.pop('saved_event',None)
                st.session_state['captured_evidence']=[]
                st.subheader('Đang theo dõi và chấm điểm')
                st.button('Dừng xử lý',key='stop_live',help='Dừng lần xử lý hiện tại; video hoàn chỉnh chỉ được lưu khi xử lý xong.')
                live_frame=st.empty();live_metrics=st.empty();live_people=st.empty()
                progress=st.progress(0.0);status=st.empty();capture_status=st.empty()
                output=ROOT/'runtime/annotated'/f'{source_key(path)}_{time.time_ns()}.mp4'
                stream=None
                try:
                    if backend=='resnet_mil':status.info('Đang nạp ResNet50. Lần đầu có thể cần tải khoảng 98 MB trọng số chính thức; các lần sau dùng cache trên máy.')
                    stream=importlib.reload(runtime_loader).get_stream_video()(path,output,pace=pace,
                        behavior=rich_behavior,use_behavior_alarm=mil_alarm,camera_id=camera,capture_evidence=auto_capture,risk_backend=backend,
                        detector_weights=detector_weights,preprocessing=preprocessing,denoise=denoise,sharpen=sharpen,
                        detector_profile=detector_profile)
                    for update in stream:
                        if update['kind']=='frame':
                            live_frame.image(update['frame'],channels='BGR',width='stretch')
                            progress.progress(update['progress'])
                            with live_metrics.container():
                                a,b,c,d=st.columns(4)
                                a.metric('Thời gian video',f"{update['timestamp']:.1f}s")
                                a_score=update['scene_score']
                                b.metric('Điểm nghi vấn CẢNH','Chờ đủ 4s' if a_score is None else f'{a_score:.3f}')
                                counts=update.get('risk_counts',{})
                                c.metric('Nghi vấn / Nguy hiểm',f"{counts.get('suspicious',0)} / {counts.get('danger',0)}")
                                d.metric('FPS xử lý thực đo',f"{update['processed_fps']:.1f}")
                                st.caption('Bộ lọc đang áp dụng: '+(', '.join(update.get('preprocessing',{}).get('applied',[])) or 'giữ ảnh gốc'))
                            if update['people']:
                                table=pd.DataFrame([dict(ID=p['track_id'],**{'Độ tin cậy người':round(p['confidence'],3),
                                    'Gần xe (giây)':p['dwell_seconds'],'Điểm CẢNH':p['scene_score'],
                                    'MIL thử nghiệm':p.get('behavior_score'),
                                    'ResNet50 thử nghiệm':p.get('visual_score'),
                                    'Mức đánh giá':p.get('risk_label'),
                                    'Lý do':'; '.join(p.get('risk_reasons',[])),
                                    'Điều kiện còn thiếu / giới hạn':'; '.join(p.get('assessment_diagnostics',[])),
                                    'Tỷ lệ khớp qua ngưỡng tin cậy':p.get('evidence',{}).get('pose_coverage'),
                                    'Tay gần vùng xe':p.get('evidence',{}).get('wrist_near_bike'),
                                    'Chuyển động tay':p.get('evidence',{}).get('hand_motion'),
                                    'Ngồi thấp/co gối (ước lượng)':p.get('evidence',{}).get('crouching_proxy') if p.get('evidence',{}).get('lower_body_valid') else None,
                                    'Góc gối trung bình (độ)':p.get('evidence',{}).get('knee_angle_degrees'),
                                    'Đủ dữ liệu đánh giá':p.get('assessment_ready'),
                                    'Cần xem xét':p['review']}) for p in update['people']])
                                live_people.dataframe(table,hide_index=True)
                            else:live_people.warning('Không phát hiện được người trong khung hình này nên chưa thể đánh giá hành vi. Không đồng nghĩa khu vực an toàn; người có thể bị che hoặc detector bỏ sót.')
                            status.caption('Ba mức là lớp quy tắc có thời gian giữ để giảm nhấp nháy; chưa phải bộ phân lớp 3 hành vi đã được kiểm định. Màu xanh không bảo đảm an toàn khi camera có điểm mù.')
                            if update.get('new_evidence'):
                                st.session_state['captured_evidence'].extend(update['new_evidence'])
                                recent=update['new_evidence'][-1]
                                capture_status.success(f"Đã tự chụp {len(st.session_state['captured_evidence'])} ảnh sự kiện. Mới nhất: #{recent['event_id']} tại {recent['video_time_seconds']:.2f}s, ID {recent['track_ids']}.")
                        elif update['kind']=='encoding':status.info('Đang lưu video có khung và điểm dưới định dạng H.264...')
                        elif update['kind']=='complete':
                            st.session_state['live_result']=update['report']
                            st.session_state['video_path']=str(path)
                    progress.progress(1.0);status.success('Hoàn tất. Có thể phát lại hoặc tải video kết quả bên dưới.')
                except Exception as e:st.error(str(e))
                finally:
                    if stream is not None:stream.close()
            else:
                try:
                    with st.spinner('Đang trích xuất đặc trưng và suy luận...'):
                        result=predict_video(path,dense=dense)
                    st.session_state['result']=result;st.session_state['video_path']=str(path)
                    st.session_state.pop('saved_event',None)
                except Exception as e:st.error(str(e))
        if is_live and 'live_result' in st.session_state:
            live=st.session_state['live_result'];annotated=Path(live['annotated_video'])
            st.caption('Detector đã dùng cho kết quả này: '+live.get('detector','Không ghi nhận'))
            if annotated.exists():
                st.subheader('Phát lại video có khung và điểm theo thời gian')
                st.video(str(annotated))
                st.caption('Video kết quả không có âm thanh. Khung người và điểm cảnh được tính khi xử lý file, chưa phải hệ thống camera RTSP.')
                st.download_button('Tải video có khung',annotated.read_bytes(),'video_theo_doi.mp4','video/mp4')
                st.download_button('Tải nhật ký theo dõi JSON',json.dumps(live,ensure_ascii=False,indent=2),'tracking.json','application/json')
                st.caption(f"Đã xử lý {live['processed_frames']} khung; {live['review_frames']} khung có ứng viên cần xem xét. Điểm cảnh cao nhất: {live['peak_scene_score']:.3f}.")
                if live.get('behavior_features'):
                    st.caption('Nguồn cảnh báo: '+live.get('alarm_mode','baseline_scene_rule')+'. Điểm hành vi hiển thị là thử nghiệm; các đặc trưng chi tiết nằm trong trường evidence của nhật ký JSON.')
                if live.get('risk_policy'):
                    policy=live['risk_policy']
                    st.caption(f"Ngưỡng điểm tham gia đánh giá: vàng {policy['yellow_score_threshold']:.2f}, đỏ {policy['red_score_threshold']:.2f}. Cần thêm điều kiện hành vi và thời gian; điểm cao riêng lẻ không đủ lên đỏ.")
                if live.get('evidence'):
                    st.subheader('Ảnh ghi nhận tự động')
                    st.caption('Mỗi ảnh ghi lại thời điểm mô hình bật dấu hiệu cần xem xét, chưa xác nhận người đó đang trộm xe. Ảnh vẫn được giữ nếu dừng xử lý trước khi hết video.')
                    for snapshot in live['evidence']:
                        with st.expander(f"Cảnh báo #{snapshot['event_id']} · {snapshot['video_time_seconds']:.2f}s · ID {snapshot['track_ids']}"):
                            st.image(snapshot['annotated_path'],caption='Ảnh có đánh dấu tại thời điểm ghi nhận',width='stretch')
                            left,right=st.columns(2)
                            left.download_button('Tải ảnh gốc PNG',Path(snapshot['original_path']).read_bytes(),f"{snapshot['id']}_original.png",'image/png',key='raw_'+snapshot['id'])
                            right.download_button('Tải ảnh đánh dấu',Path(snapshot['annotated_path']).read_bytes(),f"{snapshot['id']}_annotated.jpg",'image/jpeg',key='marked_'+snapshot['id'])
                            st.download_button('Tải metadata JSON',Path(snapshot['metadata_path']).read_bytes(),f"{snapshot['id']}.json",'application/json',key='metadata_'+snapshot['id'])
                if live['review_frames']>0 and not live.get('evidence') and 'saved_event' not in st.session_state:
                    if st.button('Lưu ứng viên cảnh báo chờ xác minh'):
                        alert_score=live.get('peak_visual_score') if live.get('alarm_mode')=='resnet_mil' else live['peak_behavior_score'] if live.get('alarm_mode')=='experimental_mil' else live['peak_scene_score']
                        eid=create_event(camera,live['video'],alert_score)
                        st.session_state['saved_event']=eid
                        folder=ROOT/'runtime/evidence';folder.mkdir(parents=True,exist_ok=True)
                        (folder/f'{eid}.json').write_text(json.dumps(live,ensure_ascii=False,indent=2),encoding='utf-8')
                        st.success(f'Đã lưu cảnh báo #{eid}; cần bảo vệ xác minh.')
        if not is_live and 'result' in st.session_state:
            r=st.session_state['result'];p=st.session_state['video_path']
            c1,c2,c3=st.columns(3)
            c1.metric('Điểm video',f"{r['video_score']:.3f}")
            c2.metric('Ngưỡng từ validation',f"{r['threshold']:.2f}")
            c3.metric('Thời gian xử lý',f"{r['elapsed_seconds']:.1f} giây")
            decision={'review':'CẦN BẢO VỆ XEM XÉT','low_score':'ĐIỂM NGHI VẤN THẤP','insufficient_quality':'KHÔNG ĐỦ CHẤT LƯỢNG'}
            st.subheader(decision[r['decision']])
            st.video(str(browser_video(p)))
            st.caption('Điểm thấp không bảo đảm an toàn. Tín hiệu chất lượng chỉ là kiểm tra sơ bộ, chưa được hiệu chỉnh theo camera.')
            df=pd.DataFrame(r['windows'])
            if r['mode']=='dense':
                st.line_chart(df.set_index('start')[['score']])
                st.caption('Mỗi điểm dùng tối đa 48 giây ngữ cảnh trước đó. Chưa đo độ chính xác định vị; đầu biểu đồ có ít ngữ cảnh hơn.')
            st.dataframe(df,hide_index=True)
            st.download_button('Tải kết quả JSON',json.dumps(r,ensure_ascii=False,indent=2),'prediction.json','application/json')
            if r['decision']=='review' and 'saved_event' not in st.session_state:
                if st.button('Lưu cảnh báo chờ xác minh'):
                    event_id=create_event(camera,r['video'],r['video_score'])
                    st.session_state['saved_event']=event_id
                    folder=ROOT/'runtime/evidence';folder.mkdir(parents=True,exist_ok=True)
                    (folder/f'{event_id}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
                    st.success(f'Đã lưu kết quả cấp video #{event_id}. Dùng chế độ theo dõi trực tiếp để tự chụp đúng khung có dấu hiệu cần xem xét.')
with tabs[1]:
    if role=='Cư dân':st.info('Demo không công khai video hoặc sổ cảnh báo cho giao diện cư dân.')
    else:
        with connect() as db:events=pd.read_sql_query('SELECT * FROM events ORDER BY id DESC',db)
        display_events=events.copy()
        if 'risk_level' in display_events:
            display_events['Mức đánh giá']=display_events['risk_level'].map(LABELS).fillna('Chưa phân mức')
            display_events=display_events.drop(columns=['risk_level'])
        st.dataframe(display_events,hide_index=True,width='stretch')
        if not events.empty:
            event_id=st.selectbox('Mã cảnh báo',events.id.tolist())
            snapshots=list_evidence(event_id)
            for snapshot in snapshots:
                st.caption(f"Ảnh tại {snapshot['video_time']:.2f}s · frame nguồn {snapshot['source_frame']} (đếm từ 0) · ID {snapshot['track_ids']}")
                marked=Path(snapshot['annotated_path']);original=Path(snapshot['original_path']);metadata=Path(snapshot['metadata_path'])
                if marked.exists():
                    st.image(str(marked),caption='Dấu hiệu nghi vấn do mô hình ghi nhận, chờ xác minh',width='stretch')
                    st.download_button('Tải ảnh có khung',marked.read_bytes(),f"{snapshot['id']}_annotated.jpg",'image/jpeg',key='event_marked_'+snapshot['id'])
                if original.exists():st.download_button('Tải ảnh gốc',original.read_bytes(),f"{snapshot['id']}_original.png",'image/png',key='event_raw_'+snapshot['id'])
                if metadata.exists():st.download_button('Tải thông tin thời điểm ghi nhận',metadata.read_bytes(),f"{snapshot['id']}.json",'application/json',key='event_json_'+snapshot['id'])
                if not marked.exists() or not original.exists():st.warning('Tệp ảnh đã bị di chuyển hoặc không còn trên máy.')
            status=st.selectbox('Kết luận của bảo vệ',['pending','confirmed','false_alarm'])
            note=st.text_area('Ghi chú xác minh')
            if st.button('Cập nhật kết luận'):
                update_event(event_id,status,note,role);st.success('Đã cập nhật và ghi nhật ký.');st.rerun()
        if role=='Quản trị':
            with connect() as db:audit=pd.read_sql_query('SELECT * FROM audit ORDER BY id DESC',db)
            st.dataframe(audit,hide_index=True)
with tabs[2]:
    behavior_metrics=ROOT/'artifacts/behavior_v2/metrics.json'
    visual_metrics=ROOT/'artifacts/resnet_v3/metrics.json'
    if visual_metrics.exists():
        vm=json.loads(visual_metrics.read_text())
        st.subheader('Thử nhánh ResNet50 từ repo tham khảo')
        st.caption('Repo Dilaniu không cung cấp checkpoint phân lớp trộm. Dự án này dùng encoder ResNet50 chính thức và huấn luyện head MIL mới trên dữ liệu hiện có; không dùng quy tắc so khoảng cách khuôn mặt của notebook.')
        st.dataframe(pd.DataFrame([{'Cấu hình':name,'F1 validation':value['validation']['f1'],
            'F1 test cũ (chẩn đoán)':value['diagnostic_test']['f1'],'AUC test cũ (chẩn đoán)':value['diagnostic_test']['roc_auc'],
            'Chọn bằng validation':name==vm['selected_model']} for name,value in vm['variants'].items()]),hide_index=True)
        st.caption('Ba màu là chính sách quy tắc theo thời gian, chưa phải mô hình phân lớp bẻ khóa/ăn trộm đã được huấn luyện và kiểm định riêng.')
    if behavior_metrics.exists():
        bm=json.loads(behavior_metrics.read_text())
        st.subheader('Thử nghiệm đặc trưng tư thế và tương tác')
        comparison=[{'Mô hình':'Baseline toàn cảnh','F1 validation':bm['original_baseline_validation']['f1'],
                     'F1 test cũ':bm['original_baseline']['f1'],'AUC test cũ':bm['original_baseline']['roc_auc']}]
        comparison += [{'Mô hình':name,'F1 validation':value['validation']['f1'],
                        'F1 test cũ':value['diagnostic_test']['f1'],'AUC test cũ':value['diagnostic_test']['roc_auc']}
                       for name,value in bm['variants'].items()]
        st.dataframe(pd.DataFrame(comparison),hide_index=True)
        st.caption('Test cũ đã được xem trong quá trình phát triển nên đây là so sánh chẩn đoán. MIL tư thế V2 chưa vượt baseline trên validation; các số của nhánh ResNet50 nằm ở bảng riêng phía trên.')
        external=ROOT/'artifacts/behavior_v2/external_evaluation.json'
        if external.exists():
            data=json.loads(external.read_text())
            st.subheader('Video đối chứng do bạn cung cấp')
            st.dataframe(pd.DataFrame([{'Video':r['source']['title'],'Điểm baseline':r['baseline_score'],
                'Điểm MIL lấy mẫu':r['behavior_score'],'Khung cần xem xét khi duyệt đầy đủ':r.get('dense_review_frames')}
                for r in data]),hide_index=True)
            st.caption('Các video này ở bãi xe ngoài trời, nhãn bình thường còn mang tính giả định; chưa có xác minh quyền lấy xe. Không dùng để huấn luyện hoặc chỉnh ngưỡng.')
    metrics=ROOT/'artifacts/metrics.json'
    if metrics.exists():
        m=json.loads(metrics.read_text());st.json(m)
        st.image(str(ROOT/'artifacts/evaluation.png'))
    else:st.info('Chạy scripts/train.py trước.')
