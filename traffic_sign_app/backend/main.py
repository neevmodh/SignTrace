"""Run: .venv312/bin/python -m uvicorn traffic_sign_app.backend.main:app --host 127.0.0.1 --port 8000"""
from contextlib import asynccontextmanager
from pathlib import Path
import os
import shutil
import uuid
import cv2
import numpy as np
from fastapi import FastAPI,File,Form,HTTPException,UploadFile
from fastapi.responses import FileResponse,Response
from fastapi.staticfiles import StaticFiles
from .vision import SignModel,ROOT,MODEL_PATH,regions_from_frame,classify_region,draw_detections
from .jobs import JobService,APP_ROOT

service=JobService()
MAX_UPLOAD=250*1024*1024


@asynccontextmanager
async def lifespan(app):
    try:
        service.model=SignModel(Path(os.environ.get('GTSRB_MODEL_PATH',str(MODEL_PATH))))
    except Exception as error:
        service.model_error=str(error)
    service.restore()
    try:service.prepare_demo()
    except Exception as error:service.demo_error=str(error)
    yield
    service.shutdown()


app=FastAPI(title='SignTrace API',version='1.0.0',lifespan=lifespan)


@app.get('/api/health')
def health():
    return dict(ready=service.model is not None,error=service.model_error,
                ffmpeg_available=shutil.which('ffmpeg') is not None,demo_ready=service.demo_ready,demo_filename=service.demo_source.name)


@app.get('/api/model')
def model_info():
    if not service.model:return dict(ready=False,error=service.model_error)
    return service.model.info


def ready():
    if service.model is None:raise HTTPException(503,service.model_error or 'Model is loading.')
    return service.model


def options(threshold,stride,ocr):
    if not .1<=threshold<=.99:raise HTTPException(422,'Confidence must be between 0.1 and 0.99.')
    if not 1<=stride<=12:raise HTTPException(422,'Stride must be between 1 and 12.')
    return dict(threshold=threshold,stride=stride,required_hits=3,ocr=ocr)


@app.get('/api/demo/{asset}')
def demo_asset(asset:str):
    names={'video':'demo_browser.mp4','poster':'demo_poster.jpg'}
    if asset not in names:raise HTTPException(404,'Unknown demo asset.')
    path=service.runtime/names[asset]
    if not path.exists():raise HTTPException(404,'The sample clip is not available.')
    return FileResponse(path,media_type='video/mp4' if asset=='video' else 'image/jpeg')


@app.get('/api/samples')
def sample_catalog():
    return [{k:v for k,v in sample.items() if k not in {'path','poster_path'}}
            for sample in service.samples.values()]


@app.get('/api/samples/{sample_id}/{asset}')
def sample_asset(sample_id:str,asset:str):
    sample=service.samples.get(sample_id)
    if sample is None or asset not in {'video','poster'}:raise HTTPException(404,'Sample not found.')
    return FileResponse(sample['path'] if asset=='video' else sample['poster_path'],
                        media_type='video/mp4' if asset=='video' else 'image/jpeg')


@app.post('/api/jobs/demo',status_code=202)
def demo_job(threshold:float=Form(.55),stride:int=Form(3),ocr:bool=Form(True),sample_id:str=Form('german-long-drive')):
    ready()
    sample=service.samples.get(sample_id)
    if sample is None:raise HTTPException(404,'Sample not found.')
    path=sample['path']
    if not path.exists():raise HTTPException(404,'The real road sample is missing.')
    try:return service.snapshot(service.create(path,path.name,options(threshold,stride,ocr)))
    except ValueError as error:raise HTTPException(429,str(error))


@app.post('/api/jobs',status_code=202)
async def upload_video(file:UploadFile=File(...),threshold:float=Form(.55),stride:int=Form(3),ocr:bool=Form(True)):
    ready();settings=options(threshold,stride,ocr)
    suffix=Path(file.filename or '').suffix.lower()
    if suffix not in {'.mp4','.mov','.avi','.mkv','.webm','.m4v'}:
        raise HTTPException(415,'Upload an MP4, MOV, AVI, MKV or WebM video.')
    incoming=service.runtime/'uploads';incoming.mkdir(exist_ok=True)
    path=incoming/(uuid.uuid4().hex+suffix);size=0
    try:
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>MAX_UPLOAD:raise HTTPException(413,'Maximum video size is 250 MB.')
                out.write(chunk)
        if size==0:raise HTTPException(422,'The uploaded file is empty.')
        return service.snapshot(service.create(path,Path(file.filename).name,settings))
    except ValueError as error:
        path.unlink(missing_ok=True);raise HTTPException(429,str(error))
    except Exception:
        path.unlink(missing_ok=True);raise
    finally:await file.close()


@app.get('/api/jobs')
def jobs():
    return sorted([service.snapshot(j) for j in list(service.jobs.values())],key=lambda j:j['created_at'],reverse=True)


def find_job(identity):
    job=service.jobs.get(identity)
    if job is None:raise HTTPException(404,'Job not found.')
    return job


@app.get('/api/jobs/{identity}')
def job_status(identity:str):return service.snapshot(find_job(identity))


@app.post('/api/jobs/{identity}/cancel')
def cancel_job(identity:str):
    job=find_job(identity);job.cancel.set();return dict(cancel_requested=True)


@app.get('/api/jobs/{identity}/preview')
def job_preview(identity:str):
    path=find_job(identity).folder/'preview.jpg'
    if not path.exists():raise HTTPException(404,'First frame is still being processed.')
    return FileResponse(path,media_type='image/jpeg',headers={'Cache-Control':'no-store'})


@app.get('/api/jobs/{identity}/video')
def job_video(identity:str,download:bool=False):
    job=find_job(identity)
    if job.state['status']!='done':raise HTTPException(409,'The video is not finished yet.')
    return FileResponse(job.folder/'result.mp4',media_type='video/mp4',
        filename=(Path(job.filename).stem+'_signtrace.mp4') if download else None)


@app.get('/api/jobs/{identity}/alerts.csv')
def job_csv(identity:str):
    job=find_job(identity);path=job.folder/'alerts.csv'
    if not path.exists():raise HTTPException(409,'Events are not ready yet.')
    return FileResponse(path,media_type='text/csv',filename='traffic_sign_alerts.csv')


@app.get('/api/jobs/{identity}/events.json')
def job_events(identity:str):
    job=find_job(identity);path=job.folder/'events.json'
    if not path.exists():raise HTTPException(409,'Events are not ready yet.')
    return FileResponse(path,media_type='application/json',filename='traffic_sign_events.json')


@app.post('/api/images')
def image_scan(file:UploadFile=File(...),threshold:float=Form(.55),ocr:bool=Form(True),cropped:bool=Form(False)):
    model=ready();raw=file.file.read(20*1024*1024+1)
    if len(raw)>20*1024*1024:raise HTTPException(413,'Maximum image size is 20 MB.')
    frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if frame is None:raise HTTPException(422,'Cannot decode this image.')
    if max(frame.shape[:2])>4096:raise HTTPException(422,'Image dimensions must not exceed 4096 pixels.')
    if not .1<=threshold<=.99:raise HTTPException(422,'Invalid confidence threshold.')
    if cropped:
        probabilities=model.predict([frame])[0];top=model.top3(probabilities)
        detections=[dict(box=[0,0,frame.shape[1],frame.shape[0]],track_id=1,**top[0],
                         source='gtsrb',uncertain=top[0]['confidence']<threshold,model_top3=top)]
    else:
        regions=regions_from_frame(frame)
        crops=[frame[y:y+h,x:x+w] for x,y,w,h in (r.box for r in regions)]
        probabilities=model.predict(crops);detections=[]
        for index,(region,crop,probs) in enumerate(zip(regions,crops,probabilities),1):
            evidence=classify_region(model,region,crop,probs,threshold,ocr)
            detections.append(dict(box=list(region.box),track_id=index,**evidence))
    annotated=draw_detections(frame.copy(),detections)
    import base64
    ok,jpeg=cv2.imencode('.jpg',annotated)
    if not ok:raise HTTPException(500,'Image rendering failed.')
    return dict(detections=detections,image='data:image/jpeg;base64,'+base64.b64encode(jpeg).decode(),
                width=frame.shape[1],height=frame.shape[0])


app.mount('/',StaticFiles(directory=APP_ROOT/'frontend',html=True),name='frontend')
