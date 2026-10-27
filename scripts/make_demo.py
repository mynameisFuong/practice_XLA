"""Procedural illustration, never training data or evidence of accuracy."""
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'data/demo';folder.mkdir(parents=True,exist_ok=True)
    for scenario in ['owner_leaves','loitering_then_move']:
        writer=cv2.VideoWriter(str(folder/f'{scenario}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(640,360))
        if not writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
        for i in range(160):
            t=i/8
            frame=np.full((360,640,3),(65,65,65),np.uint8)
            for x in [50,230,410,590]:cv2.line(frame,(x,100),(x,300),(180,180,180),2)
            cv2.rectangle(frame,(20,60),(620,320),(130,130,130),2)
            shift=int(max(0,t-12)*22)
            bx=300+shift;by=220
            for wheelx in [bx-35,bx+35]:cv2.circle(frame,(wheelx,by+25),17,(15,15,15),-1)
            cv2.line(frame,(bx-35,by+25),(bx,by),(180,80,40),9)
            cv2.line(frame,(bx,by),(bx+35,by+25),(180,80,40),9)
            cv2.line(frame,(bx-5,by-10),(bx+20,by-10),(220,120,60),9)
            cv2.line(frame,(bx+23,by-8),(bx+28,by-35),(190,190,190),4)
            px=int(100+min(t,5)*35) if t<5 else bx-10
            py=165+int(10*np.sin(t*5)) if scenario=='loitering_then_move' and 5<t<12 else 165
            cv2.circle(frame,(px,py-25),12,(50,170,230),-1)
            cv2.line(frame,(px,py-12),(px,py+27),(50,170,230),8)
            cv2.line(frame,(px,py+27),(px-15,py+55),(50,170,230),5)
            cv2.line(frame,(px,py+27),(px+15,py+55),(50,170,230),5)
            cv2.line(frame,(px,py),(bx+5,by-20),(50,170,230),5)
            cv2.putText(frame,'SYNTHETIC - PIPELINE TEST ONLY',(20,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            cv2.putText(frame,f'{scenario}  {t:04.1f}s',(20,345),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            writer.write(frame)
        writer.release()
    print('Generated 2 synthetic pipeline-only examples; no accuracy claims.')

if __name__=='__main__':main()
