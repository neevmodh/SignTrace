"""Regression checks against the supplied checkpoint and road.mp4; no fabricated accuracy."""
from pathlib import Path
import json
import shutil
import subprocess
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from traffic_sign_app.backend.vision import (SignModel,SignTracker,Region,ROOT,regions_from_frame,
                                           classify_region,overlap)
from traffic_sign_app.backend.jobs import JobService,encode_browser_video
from traffic_sign_app.backend import main


@pytest.fixture(scope='module')
def model():return SignModel(device='cpu')


def video_frame(seconds):
    cap=cv2.VideoCapture(str(ROOT/'road.mp4'));cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
    ok,frame=cap.read();cap.release();assert ok
    return frame


def test_checkpoint_is_ready(model):
    assert model.info['ready'] and model.info['classes']==43
    probabilities=model.predict([np.ones((48,48,3),np.uint8)*128])
    assert probabilities.shape==(1,43)
    assert np.isfinite(probabilities).all() and probabilities.sum()==pytest.approx(1,abs=1e-5)


@pytest.mark.parametrize('seconds,expected',[(1,'Speed limit 30 km/h'),(10,'Stop'),(13,'Speed limit 50 km/h')])
def test_actual_video_text_readings(model,seconds,expected):
    frame=video_frame(seconds);region=regions_from_frame(frame)[0]
    x,y,w,h=region.box;crop=frame[y:y+h,x:x+w]
    evidence=classify_region(model,region,crop,model.predict([crop])[0],.55,True)
    assert evidence['label']==expected and evidence['source']=='ocr'
    assert not evidence['uncertain']


def evidence(label='Stop',observed_frame=None):
    out=dict(label=label,class_id=14,confidence=.9,source='ocr',uncertain=False)
    if observed_frame is not None:out['observed_frame']=observed_frame
    return out


def test_one_vote_per_track_per_frame():
    tracker=SignTracker();track=tracker.associate([Region((20,20,40,40),'red','round',.9)],0)[0]
    assert all(tracker.record(track,evidence(),0,0) is None for _ in range(5))
    assert len(track.history)==1
    assert tracker.record(track,evidence(),1,.1) is None
    assert tracker.record(track,evidence(),2,.2)['label']=='Stop'
    assert tracker.record(track,evidence(),3,.3) is None


def test_cached_ocr_is_not_three_independent_observations():
    tracker=SignTracker();track=tracker.associate([Region((20,20,40,40),'red','round',.9)],0)[0]
    for frame in [0,3,6]:assert tracker.record(track,evidence(observed_frame=0),frame,frame/18) is None
    assert len(track.history)==1
    assert tracker.record(track,evidence(observed_frame=9),9,.5) is None
    assert tracker.record(track,evidence(observed_frame=18),18,1.)['label']=='Stop'


def test_track_association_is_one_to_one():
    t=SignTracker();a=Region((20,20,40,40),'red','round',.9)
    t.associate([a],0)
    matched=t.associate([a,Region((24,20,40,40),'red','round',.9)],1)
    assert len({track.id for track in matched})==2


def test_blank_frame_does_not_become_a_sign():
    assert regions_from_frame(np.full((240,320,3),255,np.uint8))==[]
    assert overlap((0,0,10,10),(20,20,10,10))==0


@pytest.fixture(scope='module')
def client(tmp_path_factory,model):
    original_service,original_model=main.service,main.SignModel
    main.service=JobService(tmp_path_factory.mktemp('api-runtime'))
    main.SignModel=lambda *args,**kwargs:model
    with TestClient(main.app) as instance:yield instance
    main.service=original_service;main.SignModel=original_model


def test_frontend_health_and_demo_playback(client):
    assert client.get('/').status_code==200
    assert 'SignTrace' in client.get('/').text
    assert client.get('/api/health').json()['ready']
    assert client.get('/api/health').json()['demo_filename']=='german-long-drive.mp4'
    response=client.get('/api/demo/video',headers={'Range':'bytes=0-1023'})
    assert response.status_code==206 and len(response.content)==1024
    assert response.headers['content-type']=='video/mp4'


def test_bad_uploads_and_invalid_options(client):
    assert client.post('/api/jobs',files={'file':('bad.txt',b'not a video','text/plain')}).status_code==415
    assert client.post('/api/jobs/demo',data={'threshold':2}).status_code==422
    assert client.get('/api/jobs/missing/video').status_code==404


def test_image_api_reads_the_actual_30_sign(client):
    ok,png=cv2.imencode('.png',video_frame(1));assert ok
    response=client.post('/api/images',files={'file':('frame.png',png.tobytes(),'image/png')})
    assert response.status_code==200
    data=response.json()
    assert data['image'].startswith('data:image/jpeg;base64,')
    assert any(d['label']=='Speed limit 30 km/h' and d['source']=='ocr' for d in data['detections'])


def wait_done(client,identity,timeout=120):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        data=client.get('/api/jobs/'+identity).json()
        if data['status'] in ['done','error','cancelled']:return data
        time.sleep(.15)
    pytest.fail('Video job exceeded its test timeout.')


def test_complete_video_export_keeps_every_frame(client):
    response=client.post('/api/jobs',files={'file':('road.mp4',(ROOT/'road.mp4').read_bytes(),'video/mp4')},data={'threshold':.55,'stride':3,'ocr':'true'})
    assert response.status_code==202
    identity=response.json()['id'];result=wait_done(client,identity)
    assert result['status']=='done',result.get('error')
    assert result['frame']==270 and result['duration_seconds']==15.
    labels=[e['label'] for e in result['events']]
    assert 'Speed limit 30 km/h' in labels and 'Stop' in labels and 'Speed limit 50 km/h' in labels
    # Low-confidence contour guesses must not become spoken sign events.
    assert not any(e['source']=='candidate' for e in result['events'])
    route=client.get(result['video_url'],headers={'Range':'bytes=0-1023'})
    assert route.status_code==206
    path=main.service.jobs[identity].folder/'result.mp4'
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,pix_fmt,nb_frames','-of','json',str(path)]))['streams'][0]
    assert info['codec_name']=='h264' and info['pix_fmt']=='yuv420p' and int(info['nb_frames'])==270
    assert client.get(result['csv_url']).status_code==200


def test_cancel_job(client):
    job=client.post('/api/jobs/demo').json()
    assert client.post('/api/jobs/'+job['id']+'/cancel').status_code==200
    assert wait_done(client,job['id'],10)['status']=='cancelled'


def test_invalid_video_is_reported_as_failed_job(client):
    result=client.post('/api/jobs',files={'file':('broken.mp4',b'not an actual video','video/mp4')})
    assert result.status_code==202
    job=wait_done(client,result.json()['id'],10)
    assert job['status']=='error' and 'video' in job['error'].lower()


def test_encoder_preserves_original_audio(tmp_path):
    source=tmp_path/'audio-source.mp4';destination=tmp_path/'browser.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','color=c=blue:s=160x120:r=12:d=1',
        '-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','mpeg4','-c:a','aac',str(source)],check=True)
    encode_browser_video(source,destination,original=source,duration=1.)
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type,codec_name',
                                                '-of','json',str(destination)]))['streams']
    assert any(s['codec_type']=='audio' for s in streams)
    assert any(s['codec_name']=='h264' for s in streams)
    content=destination.read_bytes();assert content.index(b'moov')<content.index(b'mdat')


def test_notebook_upload_stream_uses_real_crlf(monkeypatch,tmp_path):
    """Exercise the actual notebook upload function without starting a second server."""
    import ast
    import http.client
    import uuid
    notebook=json.loads((ROOT/'video_sign_alert.ipynb').read_text())
    for index,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']),filename=f'cell_{index}')
    tree=ast.parse(''.join(notebook['cells'][8]['source']))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef))
    sent=[];headers={}
    class Reply:
        status=202
        def read(self):return b'{"id":"notebook-test-only"}'
    class Connection:
        def __init__(self,*args,**kwargs):pass
        def putrequest(self,*args):pass
        def putheader(self,k,v):headers[k]=v
        def endheaders(self):pass
        def send(self,data):sent.append(data)
        def getresponse(self):return Reply()
        def close(self):pass
    monkeypatch.setattr(http.client,'HTTPConnection',Connection)
    namespace=dict(Path=Path,uuid=uuid,json=json,http=__import__('http'),HOST='127.0.0.1',PORT=8000,
                   CONFIDENCE=.55,FRAME_STRIDE=3,ENABLE_OCR=True)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'notebook_upload','exec'),namespace)
    sample=tmp_path/'test.mp4';sample.write_bytes(b'video-test-payload')
    assert namespace['upload_local_video'](sample)['id']=='notebook-test-only'
    body=b''.join(sent)
    assert b'\r\nContent-Disposition:' in body
    assert b'\\r\\n' not in body
    assert len(body)==headers['Content-Length']
    assert b'video-test-payload' in body


def test_real_road_building_is_not_a_sign_proposal():
    cap=cv2.VideoCapture(str(ROOT/'real_road.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,69)
    ok,frame=cap.read();cap.release()
    assert ok
    regions=regions_from_frame(frame)
    assert all(r.box[2]*r.box[3]<frame.shape[0]*frame.shape[1]*.10 for r in regions)


def test_five_distinct_samples_are_playable_and_have_posters(client):
    samples=client.get('/api/samples').json()
    assert len(samples)==5
    assert len({sample['id'] for sample in samples})==5
    paths=[]
    for sample in samples:
        assert sample['duration_seconds']>1
        assert 'path' not in sample and 'poster_path' not in sample
        response=client.get(sample['video_url'],headers={'Range':'bytes=0-1023'})
        assert response.status_code==206 and len(response.content)==1024
        assert client.get(sample['poster_url']).headers['content-type']=='image/jpeg'
        path=main.service.samples[sample['id']]['path'];paths.append(path)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,pix_fmt','-of','json',str(path)]))['streams'][0]
        assert probe['codec_name']=='h264' and probe['pix_fmt']=='yuv420p'
    assert len(set(paths))==5
    assert client.get('/api/samples/missing/video').status_code==404
    assert client.get('/api/samples/city/unknown').status_code==404
    assert client.post('/api/jobs/demo',data={'sample_id':'../../road'}).status_code==404


def test_sample_selection_routes_the_chosen_video(client,monkeypatch):
    from types import SimpleNamespace
    seen=[]
    def capture(source,filename,options):
        seen.append((source,filename))
        return SimpleNamespace(state={'id':'test-selection','filename':filename,'status':'queued'})
    monkeypatch.setattr(main.service,'create',capture)
    for sample in client.get('/api/samples').json():
        response=client.post('/api/jobs/demo',data={'sample_id':sample['id']})
        assert response.status_code==202
        assert seen[-1]==(main.service.samples[sample['id']]['path'],sample['filename'])
        assert response.json()['filename']==sample['filename']


def test_blue_sky_cannot_be_classified_as_a_speed_sign(model):
    probabilities=np.zeros(43);probabilities[5]=.99
    region=Region((0,0,80,30),'blue','board',.8)
    result=classify_region(model,region,np.full((30,80,3),180,np.uint8),probabilities,.55,False)
    assert result['source']=='candidate' and result['class_id'] is None


def test_actual_sign_localization_reads_german_30(model):
    source=ROOT/'traffic_sign_app/benchmark/candidates/german-008.mp4'
    cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,30)
    ok,frame=cap.read();cap.release();assert ok
    regions=model.localizer.detect(frame)
    crops=[frame[y:y+h,x:x+w] for x,y,w,h in (r.box for r in regions)]
    readings=[classify_region(model,r,c,p,.55,True) for r,c,p in zip(regions,crops,model.predict(crops))]
    assert any(r['label']=='Speed limit 30 km/h' and not r['uncertain'] for r in readings)


def test_long_drive_has_seven_separate_visual_references(client):
    samples=client.get('/api/samples').json()
    long=next(sample for sample in samples if sample['id']=='german-long-drive')
    assert long['duration_seconds']>240
    assert long['visible_sign_types']==7 and len(long['references'])==7
    assert len({entry['label'] for entry in long['references']})==7
    assert all(0<=entry['time_seconds']<long['duration_seconds'] for entry in long['references'])
