"""Local Streamlit demo: upload, infer, inspect, acknowledge and audit."""
from pathlib import Path
import hashlib,json,time
import cv2
import pandas as pd
import streamlit as st
from src.predict import predict_video
from src.storage import connect,create_event,update_event
from src.media import browser_video,source_key
from src.runtime_loader import get_stream_video

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
                for key in ['result','live_result','saved_event']:
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
            rich_behavior=st.checkbox('Trích xuất tư thế và tương tác người–xe',value=True)
            mil_alarm=False
            if rich_behavior:
                st.caption('Bổ sung 17 điểm tư thế, động tác cổ tay, thân người, thời gian tương tác và chuyển động người–xe; có cờ thiếu dữ liệu khi bị che. Khung hình hiển thị tối đa 4 FPS, đặc trưng hành vi cập nhật 2 lần/giây.')
                if (ROOT/'artifacts/behavior_v2/model.pt').exists():
                    mil_alarm=st.checkbox('Dùng điểm MIL hành vi cho cảnh báo thử nghiệm',value=False)
                st.caption('MIL học từ nhãn cả video, chưa có nhãn đúng/sai cho từng người. Điểm MIL chưa hiệu chuẩn và chưa được kiểm định tại hầm chung cư.')
            if mil_alarm:
                st.info('Khung đỏ dùng điểm MIL thử nghiệm, đủ quan sát tư thế, gần xe và vượt ngưỡng ở hai lần cập nhật liên tiếp. Đây chưa phải kết luận trộm.')
            else:
                st.info('Khung xanh: người được phát hiện. Khung đỏ: ứng viên cần xem xét khi người ở gần xe ≥ 2 giây và điểm CẢNH đạt ngưỡng. Điểm det là độ tin cậy phát hiện người; không phải xác suất ăn trộm.')
            st.caption('Chấm điểm cảnh chờ đủ 4 giây đầu và chỉ dùng quá khứ. Video hình vẽ có thể không được detector nhận là người; nên thử video thật.')
            pace=st.checkbox('Phát theo thời gian gốc',value=True)
            dense=True
        else:
            dense=st.checkbox('Phân tích liên tục các cửa sổ 4 giây (chậm hơn)',value=False)
        if st.button('Phân tích',type='primary'):
            if not path:
                st.error('Hãy chọn một video.')
            elif is_live:
                st.session_state.pop('live_result',None)
                st.session_state.pop('saved_event',None)
                st.subheader('Đang theo dõi và chấm điểm')
                st.button('Dừng xử lý',key='stop_live',help='Dừng lần xử lý hiện tại; video hoàn chỉnh chỉ được lưu khi xử lý xong.')
                live_frame=st.empty();live_metrics=st.empty();live_people=st.empty()
                progress=st.progress(0.0);status=st.empty()
                output=ROOT/'runtime/annotated'/f'{source_key(path)}_{time.time_ns()}.mp4'
                stream=None
                try:
                    stream=get_stream_video()(path,output,pace=pace,behavior=rich_behavior,use_behavior_alarm=mil_alarm)
                    for update in stream:
                        if update['kind']=='frame':
                            live_frame.image(update['frame'],channels='BGR',width='stretch')
                            progress.progress(update['progress'])
                            with live_metrics.container():
                                a,b,c,d=st.columns(4)
                                a.metric('Thời gian video',f"{update['timestamp']:.1f}s")
                                a_score=update['scene_score']
                                b.metric('Điểm nghi vấn CẢNH','Chờ đủ 4s' if a_score is None else f'{a_score:.3f}')
                                c.metric('Người cần xem xét',update['review_count'])
                                d.metric('FPS xử lý thực đo',f"{update['processed_fps']:.1f}")
                            if update['people']:
                                table=pd.DataFrame([dict(ID=p['track_id'],**{'Độ tin cậy người':round(p['confidence'],3),
                                    'Gần xe (giây)':p['dwell_seconds'],'Điểm CẢNH':p['scene_score'],
                                    'MIL thử nghiệm':p.get('behavior_score'),
                                    'Tỷ lệ khớp qua ngưỡng tin cậy':p.get('evidence',{}).get('pose_coverage'),
                                    'Tay gần vùng xe':p.get('evidence',{}).get('wrist_near_bike'),
                                    'Chuyển động tay':p.get('evidence',{}).get('hand_motion'),
                                    'Cần xem xét':p['review']}) for p in update['people']])
                                live_people.dataframe(table,hide_index=True)
                            else:live_people.info('Chưa phát hiện người trong khung hình hiện tại.')
                            status.caption('Khung đỏ là quy tắc thử nghiệm, chưa phải kết luận trộm. FPS xử lý có thể thấp hơn tốc độ video trên máy chậm.')
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
            if annotated.exists():
                st.subheader('Phát lại video có khung và điểm theo thời gian')
                st.video(str(annotated))
                st.caption('Video kết quả không có âm thanh. Khung người và điểm cảnh được tính khi xử lý file, chưa phải hệ thống camera RTSP.')
                st.download_button('Tải video có khung',annotated.read_bytes(),'video_theo_doi.mp4','video/mp4')
                st.download_button('Tải nhật ký theo dõi JSON',json.dumps(live,ensure_ascii=False,indent=2),'tracking.json','application/json')
                st.caption(f"Đã xử lý {live['processed_frames']} khung; {live['review_frames']} khung có ứng viên cần xem xét. Điểm cảnh cao nhất: {live['peak_scene_score']:.3f}.")
                if live.get('behavior_features'):
                    st.caption('Nguồn cảnh báo: '+live.get('alarm_mode','baseline_scene_rule')+'. Điểm hành vi hiển thị là thử nghiệm; các đặc trưng chi tiết nằm trong trường evidence của nhật ký JSON.')
                if live['review_frames']>0 and 'saved_event' not in st.session_state:
                    if st.button('Lưu ứng viên cảnh báo chờ xác minh'):
                        alert_score=live['peak_behavior_score'] if live.get('alarm_mode')=='experimental_mil' else live['peak_scene_score']
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
                    cap=cv2.VideoCapture(p);ok,frame=cap.read();cap.release()
                    if ok:cv2.imwrite(str(folder/f'{event_id}.jpg'),frame)
                    st.success(f'Đã lưu cảnh báo #{event_id}. Ảnh lưu là khung đầu video, không phải thời điểm trộm được xác nhận.')
with tabs[1]:
    if role=='Cư dân':st.info('Demo không công khai video hoặc sổ cảnh báo cho giao diện cư dân.')
    else:
        with connect() as db:events=pd.read_sql_query('SELECT * FROM events ORDER BY id DESC',db)
        st.dataframe(events,hide_index=True,width='stretch')
        if not events.empty:
            event_id=st.selectbox('Mã cảnh báo',events.id.tolist())
            status=st.selectbox('Kết luận của bảo vệ',['pending','confirmed','false_alarm'])
            note=st.text_area('Ghi chú xác minh')
            if st.button('Cập nhật kết luận'):
                update_event(event_id,status,note,role);st.success('Đã cập nhật và ghi nhật ký.');st.rerun()
        if role=='Quản trị':
            with connect() as db:audit=pd.read_sql_query('SELECT * FROM audit ORDER BY id DESC',db)
            st.dataframe(audit,hide_index=True)
with tabs[2]:
    behavior_metrics=ROOT/'artifacts/behavior_v2/metrics.json'
    if behavior_metrics.exists():
        bm=json.loads(behavior_metrics.read_text())
        st.subheader('Thử nghiệm đặc trưng tư thế và tương tác')
        comparison=[{'Mô hình':'Baseline toàn cảnh','F1 validation':bm['original_baseline_validation']['f1'],
                     'F1 test cũ':bm['original_baseline']['f1'],'AUC test cũ':bm['original_baseline']['roc_auc']}]
        comparison += [{'Mô hình':name,'F1 validation':value['validation']['f1'],
                        'F1 test cũ':value['diagnostic_test']['f1'],'AUC test cũ':value['diagnostic_test']['roc_auc']}
                       for name,value in bm['variants'].items()]
        st.dataframe(pd.DataFrame(comparison),hide_index=True)
        st.caption('Test cũ đã được xem trong quá trình phát triển nên đây là so sánh chẩn đoán, không phải đánh giá độc lập cuối cùng. Mô hình MIL chưa vượt baseline trên validation; giữ baseline làm mặc định.')
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
