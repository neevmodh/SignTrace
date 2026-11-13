"""Bounded background jobs, full-frame video export and browser-safe H.264."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass,field
from datetime import datetime,timezone
import copy
import csv
import json
import shutil
import subprocess
import threading
import time
import uuid
import cv2
import numpy as np
from .vision import ROOT,VideoVision
from .samples import SAMPLES

APP_ROOT=Path(__file__).resolve().parents[1]
RUNTIME=APP_ROOT/'runtime'


@dataclass
class Job:
    id:str
    source:Path
    folder:Path
    filename:str
    options:dict
    cancel:threading.Event=field(default_factory=threading.Event)
    state:dict=field(default_factory=dict)


def encode_browser_video(source,destination,original=None,duration=None,cancel=None):
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('FFmpeg is required. On macOS run: brew install ffmpeg')
    temporary=destination.with_name(destination.stem+'.partial.mp4')
    args=[ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(source)]
    if original:args+=['-i',str(original),'-map','0:v:0','-map','1:a:0?']
    args+=['-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart']
    if duration is not None:args+=['-t',f'{duration:.6f}']
    # Pad odd dimensions to an even H.264 frame without dropping image content.
    args+=['-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',str(temporary)]
    with subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE) as process:
        while process.poll() is None:
            if cancel and cancel.wait(.1):
                process.terminate();process.communicate(timeout=10)
                temporary.unlink(missing_ok=True)
                raise InterruptedError('Cancelled')
            if not cancel:time.sleep(.1)
        stderr=process.communicate()[1]
        if process.returncode:raise RuntimeError('Video encoding failed: '+stderr.decode(errors='replace')[-1000:])
    temporary.replace(destination)


class JobService:
    def __init__(self,runtime=RUNTIME):
        self.runtime=Path(runtime);self.runtime.mkdir(parents=True,exist_ok=True)
        self.model=None;self.model_error=None;self.jobs={};self.lock=threading.RLock()
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='video-analysis')
        self.demo_ready=False
        self.demo_source=SAMPLES[0]['path']
        self.samples={}

    def prepare_demo(self):
        # All samples have already been encoded as browser-safe H.264.
        for sample in SAMPLES:
            source=sample['path']
            if not source.exists():continue
            cap=cv2.VideoCapture(str(source))
            fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            poster=self.runtime/('sample_'+sample['id']+'.jpg')
            if sample.get('poster_seconds'):cap.set(cv2.CAP_PROP_POS_MSEC,sample['poster_seconds']*1000)
            ok,frame=cap.read();cap.release()
            if not ok or fps<=0:continue
            if not poster.exists() or poster.stat().st_mtime<source.stat().st_mtime:
                cv2.imwrite(str(poster),frame)
            self.samples[sample['id']]={**sample,'poster_path':poster,'filename':source.name,
                'duration_seconds':round(frames/fps,3),'width':width,'height':height,
                'video_url':f"/api/samples/{sample['id']}/video",
                'poster_url':f"/api/samples/{sample['id']}/poster"}
        source=self.demo_source
        if not source.exists():return
        preview=self.runtime/'demo_browser.mp4';poster=self.runtime/'demo_poster.jpg'
        if not preview.exists() or preview.stat().st_mtime<source.stat().st_mtime:
            encode_browser_video(source,preview)
        if not poster.exists() or poster.stat().st_mtime<source.stat().st_mtime:
            cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_MSEC,1000)
            ok,frame=cap.read();cap.release()
            if ok:cv2.imwrite(str(poster),frame)
        self.demo_ready=True

    def update(self,job,**fields):
        with self.lock:job.state.update(fields)

    def snapshot(self,job):
        with self.lock:return copy.deepcopy(job.state)

    def create(self,source,filename,options):
        if self.model is None:raise RuntimeError(self.model_error or 'Model is not ready.')
        with self.lock:
            if sum(j.state['status'] in ['queued','processing','encoding'] for j in self.jobs.values())>=3:
                raise ValueError('Three jobs are already queued. Wait or cancel a job first.')
            identity=uuid.uuid4().hex;folder=self.runtime/identity;folder.mkdir()
            job=Job(identity,Path(source),folder,filename,options)
            job.state=dict(id=identity,filename=filename,status='queued',progress=0.,frame=0,total_frames=0,
                events=[],detections=[],created_at=datetime.now(timezone.utc).isoformat(),
                preview_revision=0,error=None,options=options)
            self.jobs[identity]=job
        self.pool.submit(self.process,job)
        return job

    def process(self,job):
        cap=None;writer=None;started=time.monotonic()
        try:
            if job.cancel.is_set():raise InterruptedError('Cancelled')
            cap=cv2.VideoCapture(str(job.source))
            if not cap.isOpened():raise ValueError('This file could not be decoded as a video. Try MP4, MOV, AVI or WebM.')
            fps=float(cap.get(cv2.CAP_PROP_FPS));fps=fps if np.isfinite(fps) and 1<=fps<=240 else 25.
            total=max(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            vision=VideoVision(self.model,**job.options)
            self.update(job,status='processing',fps=fps,total_frames=total)
            raw=job.folder/'annotated_raw.mp4';events=[];frame_number=0
            while True:
                if job.cancel.is_set():raise InterruptedError('Cancelled')
                ok,frame=cap.read()
                if not ok:break
                if writer is None:
                    height,width=frame.shape[:2]
                    if min(height,width)<32 or max(height,width)>4096:
                        raise ValueError('Video resolution must be between 32 and 4096 pixels per side.')
                    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
                    if not writer.isOpened():raise RuntimeError('OpenCV could not open the temporary video writer.')
                    self.update(job,width=width,height=height)
                annotated,detections,new_events=vision.frame(frame,frame_number,fps)
                events.extend(new_events);writer.write(annotated)
                frame_number+=1
                if frame_number%3==0 or frame_number==1:
                    # Atomic JPEG replacement means the browser never sees half a frame.
                    preview=job.folder/'preview.partial.jpg';cv2.imwrite(str(preview),annotated)
                    preview.replace(job.folder/'preview.jpg')
                    self.update(job,frame=frame_number,progress=min(94.,94*frame_number/max(total,frame_number)),
                        events=events.copy(),detections=detections,preview_revision=frame_number,
                        elapsed_seconds=round(time.monotonic()-started,2))
            if writer:writer.release();writer=None
            if frame_number==0:raise ValueError('The video contains no decodable frames.')
            if total>0 and frame_number<total-2:
                raise ValueError(f'Decoder stopped early ({frame_number}/{total} frames). Re-encode the input with FFmpeg and retry.')
            self.update(job,status='encoding',progress=96.,frame=frame_number,events=events.copy())
            encode_browser_video(raw,job.folder/'result.mp4',original=job.source,duration=frame_number/fps,cancel=job.cancel)
            raw.unlink(missing_ok=True)
            with (job.folder/'alerts.csv').open('w',newline='',encoding='utf-8') as handle:
                fields=['time_seconds','frame','track_id','class_id','label','confidence','source','uncertain']
                out=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');out.writeheader();out.writerows(events)
            (job.folder/'events.json').write_text(json.dumps(events,indent=2))
            confirmed=[e for e in events if not e['uncertain']]
            self.update(job,status='done',progress=100.,duration_seconds=round(frame_number/fps,3),
                elapsed_seconds=round(time.monotonic()-started,2),confirmed_signs=len(confirmed),
                possible_signs=len(events)-len(confirmed),events=events,
                video_url=f'/api/jobs/{job.id}/video',csv_url=f'/api/jobs/{job.id}/alerts.csv',
                events_url=f'/api/jobs/{job.id}/events.json')
            (job.folder/'job.json').write_text(json.dumps(self.snapshot(job),indent=2))
        except InterruptedError:
            self.update(job,status='cancelled',error=None)
        except Exception as error:
            self.update(job,status='error',error=str(error))
        finally:
            if cap:cap.release()
            if writer:writer.release()
            # Persist failed/cancelled job information as well as successful results.
            (job.folder/'job.json').write_text(json.dumps(self.snapshot(job),indent=2))

    def restore(self):
        for path in self.runtime.glob('*/job.json'):
            try:
                state=json.loads(path.read_text());identity=state['id']
                if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):continue
                if state['status'] not in ['done','error','cancelled']:continue
                if state['status']=='done' and not (path.parent/'result.mp4').exists():continue
                job=Job(identity,Path(''),path.parent,state['filename'],state.get('options',{}),state=state)
                self.jobs[identity]=job
            except (ValueError,KeyError,OSError):continue

    def shutdown(self):
        with self.lock:
            for job in self.jobs.values():
                if job.state['status'] in ['queued','processing','encoding']:job.cancel.set()
        self.pool.shutdown(wait=True,cancel_futures=True)
