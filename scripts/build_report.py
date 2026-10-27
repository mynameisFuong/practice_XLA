"""Render measured results into the Vietnamese report and standalone HTML."""
from pathlib import Path
import base64,json,platform,sys
from datetime import datetime,timezone
import importlib.metadata
import markdown
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    artifacts=ROOT/'artifacts';report=ROOT/'reports'
    m=json.loads((artifacts/'metrics.json').read_text())
    manifest=pd.read_csv(ROOT/'data/manifest.csv')
    selected=m['selected_model'];metrics=m['test'][selected]
    counts=manifest.groupby(['split','label']).size()
    rows=['| Tập | Normal | Stealing | Tổng |','|---|---:|---:|---:|']
    for split in ['train','val','test']:
        n0,n1=int(counts[split,0]),int(counts[split,1]);rows.append(f'| {split} | {n0} | {n1} | {n0+n1} |')
    rows+=['',f'Tổng {len(manifest)} video thật, dung lượng nguồn {manifest.bytes.sum()/1e6:.1f} MB (đơn vị thập phân).',
           '',f'Mô hình chọn bằng validation: **{selected}**. Ngưỡng **{m["threshold"]:.2f}**. Vector đầu vào {m["feature_dimensions"]} chiều.',
           '', '| Mô hình | F1 validation | Ngưỡng validation | Precision test | Recall test | F1 test | ROC AUC test | AP test |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,values in m['test'].items():
        v=m['validation'][name]
        rows.append(f'| {name} | {v["f1"]:.3f} | {v["threshold"]:.2f} | {values["precision"]:.3f} | {values["recall"]:.3f} | {values["f1"]:.3f} | {values["roc_auc"]:.3f} | {values["average_precision"]:.3f} |')
    cm=metrics['confusion_matrix']
    rows+=['',f'Confusion matrix của mô hình chọn: TN = {cm[0][0]}, FP = {cm[0][1]}, FN = {cm[1][0]}, TP = {cm[1][1]}. Hàng là nhãn thật Normal/Stealing; cột là dự đoán Normal/Stealing.',
           '',f'Accuracy test = {metrics["accuracy"]:.3f}. Lần chạy huấn luyện ghi nhận {m["training_seconds"]:.1f} giây, có sử dụng cache đặc trưng từ các video đã xử lý; đây không phải benchmark trích xuất toàn bộ dữ liệu từ đầu.',
           '', '**Nhận xét:** Đây là số đo thật trên tập con nhỏ, không phải độ chính xác của hệ thống trong hầm. F1 test của mô hình chọn còn thấp hơn Dummy dự đoán tất cả là dương; AUC cao hơn 0,5 chỉ cho thấy khả năng xếp hạng có tín hiệu trong tập nhỏ này, chưa đủ khẳng định hiệu quả cảnh báo. Mỗi video dương bị bỏ sót làm recall thay đổi đáng kể. Cảnh và góc camera có thể là lối tắt mà mô hình đã học. Chưa chứng minh mô hình tốt hơn trong mọi điều kiện ánh sáng hoặc nhận đúng thao tác tại khóa xe. Không so sánh trực tiếp với AUC frame-level trong bài UCF-Crime gốc.',
           '', '![Confusion matrix và đường ROC từ kết quả chạy thật](../artifacts/evaluation.png)', '']
    checks=artifacts/'verification.json'
    if checks.exists():
        verification=json.loads(checks.read_text())
        rows+=['**Kiểm chứng thực thi:** '+verification['summary'],'']
    # Both a real held-out video and a synthetic clip, kept separate.
    for filename,label in [('real_video_prediction.json','Ví dụ video thật thuộc test'),('demo_prediction.json','Ví dụ video hình vẽ kiểm tra pipeline')]:
        path=artifacts/filename
        if path.exists():
            p=json.loads(path.read_text())
            rows += [f'**{label}:** `{p["video"]}`, điểm {p["video_score"]:.3f}, quyết định `{p["decision"]}`, {len(p["windows"])} cửa sổ, thời gian {p["elapsed_seconds"]:.2f} giây.']
            if filename.startswith('demo'):rows+=['Đây không phải kết quả nhận dạng đúng/sai trên dữ liệu thật.']
            rows+=['']
    template=(report/'BAO_CAO.md').read_text(encoding='utf-8')
    if '<!-- GENERATED_RESULTS -->' in template:
        template=template.replace('<!-- GENERATED_RESULTS -->','<!-- RESULTS_BEGIN -->\n'+'\n'.join(rows)+'\n<!-- RESULTS_END -->')
    else:
        import re
        template=re.sub(r'<!-- RESULTS_BEGIN -->.*?<!-- RESULTS_END -->',lambda _:'<!-- RESULTS_BEGIN -->\n'+'\n'.join(rows)+'\n<!-- RESULTS_END -->',template,flags=re.S)
    (report/'BAO_CAO.md').write_text(template,encoding='utf-8')
    body=markdown.markdown(template,extensions=['tables','fenced_code','toc'])
    encoded=base64.b64encode((artifacts/'evaluation.png').read_bytes()).decode()
    body=body.replace('../artifacts/evaluation.png','data:image/png;base64,'+encoded)
    html='''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Báo cáo hệ thống cảnh báo trộm xe máy</title><style>
body{font-family:Arial,"Helvetica Neue",sans-serif;color:#17202a;line-height:1.65;max-width:1000px;margin:48px auto;padding:0 32px}h1{font-size:30px;line-height:1.25}h2{font-size:23px;margin-top:38px;border-bottom:1px solid #dce1e6;padding-bottom:6px}h3{font-size:18px;margin-top:24px}p{text-align:justify}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{border:1px solid #ccd3da;padding:10px;text-align:left;vertical-align:top}th{background:#eef2f5}tr:nth-child(even){background:#fafbfc}pre{background:#f2f4f6;padding:16px;overflow:auto;font-size:13px}code{font-family:ui-monospace,Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}img{display:block;max-width:100%;height:auto;margin:20px auto}a{color:#145080;overflow-wrap:anywhere}@media print{body{margin:0;max-width:none;font-size:10.5pt;padding:0}h1{font-size:22pt}h2{font-size:17pt}h3{font-size:13pt}h1,h2,h3{break-after:avoid}table,pre,img{break-inside:avoid}th,td{padding:6px;font-size:9pt}@page{size:A4;margin:18mm}}
</style></head><body>'''+body+'</body></html>'
    (report/'BAO_CAO.html').write_text(html,encoding='utf-8')
    env=dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','opencv-python','scikit-learn','streamlit','pandas']},timestamp_utc=datetime.now(timezone.utc).isoformat())
    (artifacts/'environment.json').write_text(json.dumps(env,indent=2))
    print('Built reports/BAO_CAO.html and updated measured results in Markdown')

if __name__=='__main__':main()
