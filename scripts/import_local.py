"""Convert an explicitly labeled, consented local dataset to the training schema.

Input CSV: path,label,split,session_id,camera_id. Split entire recording sessions
and cameras beforehand; never create labels from filename substrings.
"""
from pathlib import Path
import argparse,hashlib
import pandas as pd

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--out',default='data/local_manifest.csv');a=p.parse_args()
    df=pd.read_csv(a.csv)
    required={'path','label','split','session_id','camera_id'}
    if not required.issubset(df.columns):raise ValueError('Required: '+','.join(sorted(required)))
    if not set(df.label).issubset({0,1}):raise ValueError('label must be 0 or 1')
    if not set(df.split).issubset({'train','val','test'}):raise ValueError('Invalid split')
    for key in ['session_id','camera_id']:
        if df.groupby(key).split.nunique().max()>1:raise ValueError(key+' overlaps splits')
    rows=[]
    for r in df.to_dict('records'):
        path=Path(r.pop('path')).expanduser().resolve()
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(**r,video_id=sha[:16],sha256=sha,local_path=str(path),original_split='local'))
    result=pd.DataFrame(rows)
    if result.sha256.duplicated().any():raise ValueError('Duplicate file contents')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);result.to_csv(out,index=False)
    print('Saved',len(result),'videos to',out)
