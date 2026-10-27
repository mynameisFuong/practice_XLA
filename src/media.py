"""Browser-compatible H.264 playback, independent of the source codec."""
from pathlib import Path
import hashlib
import os
import subprocess
import tempfile
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]

def source_key(path):
    path=Path(path).resolve();stat=path.stat()
    return hashlib.sha256(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:24]

def encode_h264(source,target):
    """Atomic encode. Distinct temp files let concurrent sessions share a cache."""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(suffix='.mp4',dir=target.parent);os.close(fd)
    try:
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
                 '-i',str(source),'-vf',r'scale=w=trunc(min(1280\,iw)/2)*2:h=-2',
                 '-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p',
                 '-c:a','aac','-movflags','+faststart',temp]
        result=subprocess.run(command,capture_output=True,text=True,timeout=1800)
        if result.returncode:
            raise RuntimeError('Không chuyển được video sang H.264: '+result.stderr[-1500:])
        if Path(temp).stat().st_size==0:raise RuntimeError('Video chuyển đổi rỗng')
        os.replace(temp,target)
    finally:
        Path(temp).unlink(missing_ok=True)
    return target

def browser_video(source,cache_dir=None):
    folder=Path(cache_dir or ROOT/'runtime/playback')
    target=folder/(source_key(source)+'_h264_v1.mp4')
    if not target.exists():encode_h264(source,target)
    return target
