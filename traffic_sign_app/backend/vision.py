"""Use the existing GTSRB checkpoint; keep classification and OCR evidence explicit."""
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import shutil
import threading
import re
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import pytesseract
from pytesseract import Output

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'gtsrb_best_model.pth'
SPEED_CLASSES = {20: 0, 30: 1, 50: 2, 60: 3, 70: 4, 80: 5, 100: 7, 120: 8}
WARNING_CLASSES = {11, *range(18, 32)}
BLUE_CLASSES = set(range(33, 41))


def overlap(a, b):
    x, y, w, h = a; xx, yy, ww, hh = b
    inter = max(0, min(x+w, xx+ww)-max(x, xx))*max(0, min(y+h, yy+hh)-max(y, yy))
    return inter/max(1, w*h+ww*hh-inter)


@dataclass
class Region:
    box: tuple
    color: str
    shape: str
    quality: float
    localized: bool=False


def regions_from_frame(frame, min_size=16, max_regions=12):
    """Color/geometry proposals, not a trained object detector.

    Separate red/blue/green masks avoid merging neighboring sign colors. Large
    vegetation blobs and tiny roadside noise are filtered before classification.
    """
    height, width = frame.shape[:2]
    scale = min(1., 1120/width)
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0,70,55), (14,255,255)) | cv2.inRange(hsv, (162,70,55),(179,255,255))
    masks = {'red': red, 'blue': cv2.inRange(hsv,(90,65,45),(135,255,255)),
             'green':cv2.inRange(hsv,(35,65,45),(87,255,235))}
    result=[]
    for color, mask in masks.items():
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            # Broad facades and vehicles can share sign colors; reject scene-sized blobs.
            if w*h>small.shape[0]*small.shape[1]*.08:continue
            if min(w,h)<min_size*scale or w>small.shape[1]*.55 or h>small.shape[0]*.72:
                continue
            area=cv2.contourArea(contour)
            if area<100*scale*scale or area/(w*h)<.20:continue
            perimeter=cv2.arcLength(contour,True)
            polygon=cv2.approxPolyDP(contour,.035*perimeter,True)
            vertices=len(polygon);aspect=w/h
            circularity=4*np.pi*area/max(1,perimeter*perimeter)
            if color=='red':
                if not .55<=aspect<=1.65:continue
                shape='triangle' if vertices==3 else ('round' if circularity>.55 else 'polygon')
                # Red shop awnings/window frames are not octagonal or circular signs.
                if shape=='polygon' and (vertices<5 or circularity<.35):continue
                quality=.95 if shape in ['triangle','round'] else .7
            else:
                if color=='green' and not (vertices==4 and area/(w*h)>.68 and 1.2<aspect<6):continue
                if color=='blue' and not .55<aspect<6:continue
                # Blue sky and car reflections must not be treated as round arrows.
                shape='round' if color=='blue' and .70<aspect<1.35 and circularity>.62 and vertices>=5 else 'board'
                quality=.85
            # Retain a tight sign crop: the previous 12% padding included too much background.
            pad=max(2,round(.035*max(w,h)))
            x0=max(0,int((x-pad)/scale));y0=max(0,int((y-pad)/scale))
            x1=min(width,int((x+w+pad)/scale));y1=min(height,int((y+h+pad)/scale))
            result.append(Region((x0,y0,x1-x0,y1-y0),color,shape,quality))
    result.sort(key=lambda r:r.box[2]*r.box[3],reverse=True)
    selected=[]
    for region in result:
        if any(overlap(region.box,previous.box)>.35 for previous in selected):continue
        selected.append(region)
        if len(selected)==max_regions:break
    return selected


class SignModel:
    def __init__(self, path=MODEL_PATH, device=None):
        self.path=Path(path)
        torch.set_num_threads(min(4,torch.get_num_threads()))
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else
                                  'mps' if torch.backends.mps.is_available() else 'cpu'))
        checkpoint=torch.load(self.path,map_location='cpu',weights_only=True)
        required={'model_state_dict','model_name','num_classes','class_names','image_size','mean','std'}
        if not required.issubset(checkpoint):raise ValueError('Checkpoint metadata is incomplete.')
        count=int(checkpoint['num_classes']);self.labels=list(checkpoint['class_names'])
        if count!=43 or len(self.labels)!=count:raise ValueError('Expected the trained 43-class GTSRB checkpoint.')
        name=checkpoint['model_name']
        if name=='mobilenet_v3_large':
            net=models.mobilenet_v3_large(weights=None)
            net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,count)
        elif name=='resnet18':
            net=models.resnet18(weights=None)
            net.fc=nn.Sequential(nn.Linear(net.fc.in_features,256),nn.ReLU(inplace=True),nn.Dropout(.3),nn.Linear(256,count))
        else:raise ValueError(f'Unsupported checkpoint architecture: {name}')
        net.load_state_dict(checkpoint['model_state_dict'],strict=True)
        if any(not torch.isfinite(p).all() for p in net.parameters()):raise ValueError('Nonfinite model weights.')
        self.net=net.to(self.device).eval()
        self.transform=transforms.Compose([transforms.Resize((int(checkpoint['image_size']),)*2),
            transforms.ToTensor(),transforms.Normalize(checkpoint['mean'],checkpoint['std'])])
        self.lock=threading.Lock()
        self.ocr_available=shutil.which('tesseract') is not None
        from .localizer import SignLocalizer
        detector_dir=ROOT/'traffic_sign_app/models'
        self.localizer=SignLocalizer(detector_dir) if (detector_dir/'detection.pb').exists() else None
        self.info=dict(ready=True,architecture=name,classes=count,device=str(self.device),
            checkpoint=str(self.path),checkpoint_name=self.path.name,
            validation_accuracy=checkpoint.get('best_val_accuracy'),
            validation_note='Stored training validation result; not video detection accuracy.',
            epoch=checkpoint.get('epoch'),stage=checkpoint.get('stage'),image_size=checkpoint['image_size'],
            ocr_available=self.ocr_available,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            localization='Tiled GTSDB SSD sign detector with spatial tracking' if self.localizer else 'Color and geometry proposals with spatial tracking',class_names=self.labels,
            localizer_available=self.localizer is not None)
        # A real forward pass verifies architecture, weights and preprocessing compatibility.
        probe=self.predict([np.full((64,64,3),128,np.uint8)])[0]
        if probe.shape!=(43,) or not np.isfinite(probe).all():raise ValueError('Invalid model forward output.')

    def predict(self, crops):
        if not crops:return np.empty((0,43))
        tensors=[self.transform(Image.fromarray(cv2.cvtColor(c,cv2.COLOR_BGR2RGB))) for c in crops]
        with self.lock,torch.inference_mode():
            return self.net(torch.stack(tensors).to(self.device)).softmax(dim=1).cpu().numpy()

    def top3(self, probabilities):
        return [dict(class_id=int(i),label=self.labels[i],confidence=float(probabilities[i]))
                for i in np.argsort(probabilities)[-3:][::-1]]


def read_text(crop, region, enabled=True):
    """Read actual crop pixels. Never infer text from video timestamps or filenames."""
    if not enabled or shutil.which('tesseract') is None:return None
    if region.color=='blue' and region.shape=='round':return None
    if min(crop.shape[:2])<18:return None
    size=max(2,min(5,320//max(1,crop.shape[0])))
    enlarged=cv2.resize(crop,None,fx=size,fy=size,interpolation=cv2.INTER_CUBIC)
    rgb=cv2.cvtColor(enlarged,cv2.COLOR_BGR2RGB)
    if region.color=='red' and region.shape!='triangle':
        # Isolate dark digits inside the white disk. Reading the whole red ring
        # as one word made Tesseract miss "30" and "50" in the supplied clip.
        h,w=enlarged.shape[:2]
        hsv=cv2.cvtColor(enlarged,cv2.COLOR_BGR2HSV)
        gray=cv2.cvtColor(enlarged,cv2.COLOR_BGR2GRAY)
        yy,xx=np.ogrid[:h,:w]
        inside=(xx-w/2)**2+(yy-h/2)**2<(min(h,w)*.36)**2
        ink=(gray<130)&(hsv[:,:,1]<135)&inside
        ys,xs=np.where(ink)
        if len(xs)>max(12,h*w*.002):
            digits=np.where(ink,0,255).astype(np.uint8)
            digits=digits[max(0,ys.min()-12):ys.max()+13,max(0,xs.min()-12):xs.max()+13]
            try:
                data=pytesseract.image_to_data(digits,config='--psm 8 -c tessedit_char_whitelist=0123456789',
                                               output_type=Output.DICT,timeout=2)
                for text,conf in zip(data['text'],data['conf']):
                    if re.fullmatch('(20|30|50|60|70|80|100|120)',text.strip()) and float(conf)>=50:
                        speed=int(text.strip())
                        return dict(class_id=SPEED_CLASSES[speed],label=f'Speed limit {speed} km/h',
                                    confidence=float(conf)/100,source='ocr',uncertain=False,text=str(speed))
            except (RuntimeError,pytesseract.TesseractError):pass
    variants=[rgb]
    if region.color=='red' and region.shape!='triangle':
        # Inset removes the red ring and surrounding green foliage from numeric OCR.
        h,w=rgb.shape[:2];margin=int(min(h,w)*.19)
        inner=rgb[margin:h-margin,margin:w-margin]
        gray=cv2.cvtColor(inner,cv2.COLOR_RGB2GRAY)
        variants.append(cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
    readings=[]
    for image in variants:
        try:
            data=pytesseract.image_to_data(image,config='--psm 6',output_type=Output.DICT,timeout=2)
        except (RuntimeError,pytesseract.TesseractError):continue
        words=[(t.strip(),float(c)) for t,c in zip(data['text'],data['conf']) if t.strip() and float(c)>=20]
        if words:
            text=' '.join(t for t,_ in words);confidence=float(np.mean([c for _,c in words]))/100
            readings.append((text,confidence))
    for text,confidence in readings:
        clean=re.sub('[^A-Z0-9 ]','',text.upper()).strip()
        if region.color=='red' and re.search(r'\bSTOP\b',clean) and confidence>=.35:
            return dict(class_id=14,label='Stop',confidence=confidence,source='ocr',uncertain=False,text='STOP')
        if region.color=='red' and region.shape!='triangle':
            match=re.fullmatch(r'(20|30|50|60|70|80|100|120)',clean.replace(' ',''))
            if match and confidence>=.35:
                speed=int(match.group())
                return dict(class_id=SPEED_CLASSES[speed],label=f'Speed limit {speed} km/h',
                            confidence=confidence,source='ocr',uncertain=False,text=str(speed))
        if region.shape=='board' and confidence>=.55 and len(clean)>=3 and any(c.isalpha() for c in clean):
            return dict(class_id=None,label=text[:100],confidence=confidence,source='ocr',uncertain=False,text=text[:100])
    return None


def classify_region(model, region, crop, probabilities, threshold, use_ocr=True):
    top3=model.top3(probabilities)
    reading=read_text(crop,region,use_ocr)
    if reading:
        return {**reading,'model_top3':top3}
    # Geometry can reject incompatible classes without inflating model confidence.
    allowed=({*WARNING_CLASSES,13} if region.color=='red' and region.shape=='triangle' else
             BLUE_CLASSES if region.color=='blue' and region.shape=='round' else
             {*range(11),14,15,16,17} if region.color=='red' else set())
    if not allowed:
        return dict(class_id=None,label='Uncertain sign',confidence=float(max(probabilities)),
                    source='candidate',uncertain=True,text=None,model_top3=top3)
    index=max(allowed,key=lambda i:float(probabilities[i]))
    confidence=float(probabilities[index])
    if (not region.localized and region.color=='red' and region.shape!='triangle'
        and index in SPEED_CLASSES.values() and confidence<.85):
        return dict(class_id=None,label='Uncertain speed sign',confidence=confidence,
                    source='candidate',uncertain=True,text=None,model_top3=top3)
    if confidence>=threshold:
        return dict(class_id=index,label=model.labels[index],confidence=confidence,
                    source='gtsrb',uncertain=False,text=None,model_top3=top3)
    if confidence>=.35 and region.color=='red' and region.shape=='triangle':
        return dict(class_id=index,label='Possible '+model.labels[index],confidence=confidence,
                    source='gtsrb',uncertain=True,text=None,model_top3=top3)
    if region.color=='red' and region.shape=='triangle':
        return dict(class_id=None,label='Warning sign, symbol unclear',confidence=confidence,
                source='candidate',uncertain=True,text=None,model_top3=top3)
    return dict(class_id=None,label='Uncertain sign',confidence=confidence,
                source='candidate',uncertain=True,text=None,model_top3=top3)


@dataclass
class Track:
    id:int
    box:tuple
    color:str
    shape:str
    last_seen:int
    history:deque=field(default_factory=lambda:deque(maxlen=7))
    evidence:dict=field(default_factory=dict)
    confirmed:dict=field(default_factory=dict)
    last_emitted:str=''
    last_ocr_time:float=-100.
    cached_ocr:dict|None=None


class SignTracker:
    def __init__(self,required_hits=3,max_missed=10):
        self.required_hits=required_hits;self.max_missed=max_missed;self.tracks=[];self.next_id=1

    def associate(self,regions,frame_number):
        self.tracks=[t for t in self.tracks if frame_number-t.last_seen<=self.max_missed]
        pairs=[]
        for i,region in enumerate(regions):
            for track in self.tracks:
                if track.color!=region.color or (track.shape=='triangle')!=(region.shape=='triangle'):continue
                score=overlap(region.box,track.box)
                if score>=.13:pairs.append((score,i,track.id))
        matches={};used=set()
        for score,index,track_id in sorted(pairs,reverse=True):
            if index in matches or track_id in used:continue
            matches[index]=next(t for t in self.tracks if t.id==track_id);used.add(track_id)
        output=[]
        for i,region in enumerate(regions):
            track=matches.get(i)
            if track is None:
                track=Track(self.next_id,region.box,region.color,region.shape,frame_number)
                self.next_id+=1;self.tracks.append(track)
            track.box=region.box;track.last_seen=frame_number
            output.append(track)
        return output

    def record(self,track,evidence,frame_number,seconds):
        track.evidence=evidence
        # A track can gain at most ONE vote per actual analyzed video frame.
        if track.history and track.history[-1][0]==frame_number:return None
        if evidence['source']=='ocr' and any(
            entry[2].get('observed_frame')==evidence.get('observed_frame') and entry[2]['source']=='ocr'
            for entry in track.history) and 'observed_frame' in evidence:
            return None
        track.history.append((frame_number,evidence['label'],evidence))
        eligible=[entry for entry in track.history if entry[2]['source']!='candidate' or entry[2].get('readable_candidate')]
        if not eligible:return None
        label,hits=Counter(e[1] for e in eligible).most_common(1)[0]
        if hits<self.required_hits:return None
        agreeing=[e[2] for e in eligible if e[1]==label]
        current=agreeing[-1]
        if evidence['label']!=label:return None
        track.confirmed={**current,'confidence':float(np.mean([e['confidence'] for e in agreeing]))}
        if track.last_emitted==label:return None
        track.last_emitted=label
        return dict(time_seconds=round(seconds,3),frame=frame_number,track_id=track.id,
                    box=list(track.box),**track.confirmed)


def draw_detections(frame,detections,seconds=None):
    for d in detections:
        x,y,w,h=map(int,d['box'])
        uncertain=d.get('uncertain',True)
        color=(45,190,245) if uncertain else (145,215,40)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2,cv2.LINE_AA)
        text=f"#{d.get('track_id','-')} {d['label']}"
        if d.get('source')!='candidate':text+=f" {d['confidence']:.0%}"
        # Wrap long board readings instead of drawing outside the video.
        scale=max(.42,min(.65,frame.shape[1]/1600))
        while cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0][0]>frame.shape[1]-20 and len(text)>20:
            text=text[:-4]+'...'
        (tw,th),base=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
        xx=max(5,min(x,frame.shape[1]-tw-12));yy=max(th+12,y-9)
        cv2.rectangle(frame,(xx-4,yy-th-7),(xx+tw+6,yy+base+5),(20,30,40),-1)
        cv2.putText(frame,text,(xx,yy),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    if seconds is not None:
        cv2.putText(frame,f'SIGNTRACE  |  {seconds:.1f}s',(14,frame.shape[0]-16),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(230,240,245),1,cv2.LINE_AA)
    return frame


class VideoVision:
    def __init__(self,model,threshold=.55,stride=3,required_hits=3,ocr=True):
        self.model=model;self.threshold=threshold;self.stride=max(1,stride);self.ocr=ocr
        self.tracker=SignTracker(required_hits,max_missed=12)
        self.last_regions=[]

    def frame(self,frame,frame_number,fps):
        if getattr(self.model,'localizer',None):
            if frame_number%self.stride==0:self.last_regions=self.model.localizer.detect(frame)
            regions=self.last_regions
        else:regions=regions_from_frame(frame)
        tracks=self.tracker.associate(regions,frame_number)
        events=[];seconds=frame_number/fps
        if frame_number%self.stride==0 and regions:
            crops=[frame[y:y+h,x:x+w] for x,y,w,h in (r.box for r in regions)]
            probabilities=self.model.predict(crops)
            for region,track,crop,probs in zip(regions,tracks,crops,probabilities):
                # OCR is refreshed at least twice per second per visible sign.
                do_ocr=self.ocr and (seconds-track.last_ocr_time>=.45)
                evidence=classify_region(self.model,region,crop,probs,self.threshold,do_ocr)
                if do_ocr:
                    track.last_ocr_time=seconds
                    if evidence['source']=='ocr':evidence['observed_frame']=frame_number
                    track.cached_ocr=evidence if evidence['source']=='ocr' else None
                elif track.cached_ocr:
                    evidence={**track.cached_ocr,'model_top3':self.model.top3(probs)}
                event=self.tracker.record(track,evidence,frame_number,seconds)
                if event:events.append(event)
        detections=[]
        for track in tracks:
            evidence=track.confirmed or track.evidence
            if not evidence:
                evidence=dict(label='Reading sign…',confidence=0.,source='candidate',uncertain=True)
            detections.append(dict(box=list(track.box),track_id=track.id,confirmed=bool(track.confirmed),**evidence))
        return draw_detections(frame.copy(),detections,seconds),detections,events
