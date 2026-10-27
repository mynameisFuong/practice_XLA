"""Download a reproducible, bounded UCF-Crime subset; never invent labels."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import random
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = 'mteb/ucf-crime'
REV = '8d783bb'
BASE = f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/'

def download(remote, local, expected=None):
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists() or (expected and local.stat().st_size != expected):
        temp = local.with_suffix(local.suffix + '.part')
        for attempt in range(3):
            try:
                with requests.get(BASE + remote, stream=True, timeout=(30, 180)) as r:
                    r.raise_for_status()
                    with temp.open('wb') as f:
                        for chunk in r.iter_content(1024 * 1024):
                            f.write(chunk)
                if expected and temp.stat().st_size != expected:
                    raise ValueError('Incomplete download: ' + remote)
                temp.replace(local)
                break
            except Exception:
                if attempt == 2:
                    raise
    return hashlib.sha256(local.read_bytes()).hexdigest()

def main():
    data = ROOT / 'data'
    tables = {}
    for split in ['train', 'test']:
        p = data / f'{split}.parquet'
        download(f'data/{split}-00000-of-00001.parquet', p)
        tables[split] = pd.read_parquet(p)
    print('Metadata:', {k: v.category.value_counts().to_dict() for k,v in tables.items()}, flush=True)
    manifest = data / 'manifest.csv'
    if manifest.exists():
        rows = pd.read_csv(manifest).to_dict('records')
    else:
        catalog = {}
        for folder in ['Stealing','Training_Normal','Testing_Normal']:
            url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/videos/{folder}?limit=1000'
            r = requests.get(url, timeout=60); r.raise_for_status()
            for item in r.json():
                if item['type'] == 'file':
                    catalog[Path(item['path']).name] = item
        rng = random.Random(42)
        rows = []
        for original_split, table in tables.items():
            for label, category in [(0, 'Normal'), (1,'Stealing')]:
                candidates = []
                for row in table.to_dict('records'):
                    if (category == 'Normal' and 'Normal' in row['category']) or row['category'] == category:
                        item = catalog.get(Path(row['video_path']).name)
                        if item and (item['size'] <= 30_000_000 or (original_split=='test' and label==1)):
                            candidates.append((row, item))
                candidates.sort(key=lambda x: x[1]['path'])
                rng.shuffle(candidates)
                count = 25 if original_split == 'train' else 8
                for i, (meta, item) in enumerate(candidates[:count]):
                    split = ('val' if i >= 20 else 'train') if original_split == 'train' else 'test'
                    rows.append(dict(video_id=Path(item['path']).stem, label=label, split=split,
                                     original_split=original_split, remote_path=item['path'],
                                     local_path='data/videos/' + Path(item['path']).name,
                                     bytes=item['size'], source_url=BASE + item['path'],
                                     annotation_start1=meta.get('annotation_start1',-1),
                                     annotation_end1=meta.get('annotation_end1',-1),
                                     annotation_start2=meta.get('annotation_start2',-1),
                                     annotation_end2=meta.get('annotation_end2',-1)))
        pd.DataFrame(rows).to_csv(manifest,index=False)
    def task(row):
        row['sha256'] = download(row['remote_path'], ROOT / row['local_path'], int(row['bytes']))
        print('Downloaded', row['video_id'], row['split'], flush=True)
        return row
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        completed = list(pool.map(task, rows))
    pd.DataFrame(completed).to_csv(manifest,index=False)
    (data/'provenance.json').write_text(json.dumps(dict(repo=REPO, revision=REV, seed=42,
         selection='random by class within official split; <=30 MB except all test Stealing allowed; train20 val5 test<=8 per class',
         license='Upstream mirror has no explicit license grant. Research provenance retained; raw videos excluded from submission archive.'),indent=2))
    print(pd.DataFrame(completed).groupby(['split','label']).size(),flush=True)

if __name__ == '__main__':
    main()
