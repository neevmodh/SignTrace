"""Published GTSDB SSD graph for sign boxes; the user's model supplies class labels."""
from pathlib import Path
import threading
import cv2
import numpy as np


class SignLocalizer:
    def __init__(self,directory):
        directory=Path(directory)
        self.net=cv2.dnn.readNetFromTensorflow(str(directory/'detection.pb'),str(directory/'detection.pbtxt'))
        self.lock=threading.Lock()

    def detect(self,frame):
        from .vision import Region,regions_from_frame,read_text
        height,width=frame.shape[:2]
        # Small, overlapping tiles preserve distant signs before SSD's 300px resize.
        side=min(min(height,width),max(256,round(min(height,width)*.5)))
        step=max(1,round(side*.75))
        xs=sorted({*range(0,max(1,width-side+1),step),max(0,width-side)})
        ys=sorted({*range(0,max(1,height-side+1),step),max(0,height-side)})
        views=[(0,0,width,height)]+[(x,y,side,side) for y in ys for x in xs]
        proposals=[]
        with self.lock:
            for ox,oy,vw,vh in views:
                crop=frame[oy:oy+vh,ox:ox+vw]
                self.net.setInput(cv2.dnn.blobFromImage(crop,1.,(300,300),swapRB=True))
                for row in self.net.forward().reshape(-1,7):
                    confidence=float(row[2]);category=int(row[1])
                    if confidence<.80 or category not in {1,2,3}:continue
                    x0,y0,x1,y1=(row[3:]*[vw,vh,vw,vh]).astype(int)+[ox,oy,ox,oy]
                    w,h=x1-x0,y1-y0
                    if min(w,h)<10 or w*h>width*height*.12:continue
                    # Reject detections truncated at a tile boundary; other tiles cover it.
                    if (vw!=width or vh!=height) and (x0<=ox+1 or y0<=oy+1 or x1>=ox+vw-1 or y1>=oy+vh-1):continue
                    pad=max(1,round(max(w,h)*.04))
                    x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(width,x1+pad);y1=min(height,y1+pad)
                    proposals.append((confidence,category,(int(x0),int(y0),int(x1-x0),int(y1-y0))))
        # GTSDB's three detection groups omit STOP. Admit an additional filled-red
        # compact proposal only when its pixels independently read STOP via OCR.
        for region in regions_from_frame(frame):
            if region.color!='red' or region.shape=='triangle':continue
            x,y,w,h=region.box
            if not .75<w/h<1.3:continue
            crop=frame[y:y+h,x:x+w];hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            red=((hsv[:,:,0]<15)|(hsv[:,:,0]>160))&(hsv[:,:,1]>65)&(hsv[:,:,2]>55)
            if np.mean(red)<.50:continue
            reading=read_text(crop,region)
            if reading and reading.get('label')=='Stop':proposals.append((reading['confidence'],1,region.box))
        if not proposals:return []
        keep=cv2.dnn.NMSBoxes([p[2] for p in proposals],[p[0] for p in proposals],.35,.35)
        regions=[]
        for index in np.asarray(keep).reshape(-1)[:12]:
            confidence,category,box=proposals[int(index)]
            regions.append(Region(box,'blue' if category==2 else 'red',
                                  'triangle' if category==3 else 'round',confidence,localized=True))
        return regions
