"""Build the hand-in archive without virtualenv, raw CCTV or runtime evidence."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    target=ROOT/'BAI_NOP_CANH_BAO_TROM_XE.zip'
    entries=[ROOT/'README.md',ROOT/'requirements.txt',ROOT/'requirements-report.txt',ROOT/'requirements-lock.txt',ROOT/'app.py',ROOT/'.gitignore']
    for folder in ['src','scripts','tests','reports','artifacts','model','data/features','data/behavior_features','data/visual_features','data/demo']:
        entries += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    entries += [ROOT/'data'/name for name in ['manifest.csv','provenance.json','train.parquet','test.parquet']]
    entries.append(ROOT/'data/action_annotations_template.csv')
    entries += list((ROOT/'data/external_test').glob('*.json'))
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(entries)):
            if path.is_file():z.write(path,Path('CANH_BAO_TROM_XE')/path.relative_to(ROOT))
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad:raise ValueError('Archive corruption: '+bad)
        print('Archive entries:',len(z.infolist()))
    print(target,round(target.stat().st_size/1e6,2),'MB')

if __name__=='__main__':main()
