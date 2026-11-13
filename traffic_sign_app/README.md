# SignTrace — local traffic sign frontend + backend

The app uses the exact checkpoint at `/Users/neev/Downloads/DL/gtsrb_best_model.pth`. It loads the saved architecture, 43 class names, input size and normalization. It does not retrain or overwrite the model.

## Run

From `/Users/neev/Downloads/DL`:

```bash
./run_signtrace.sh
```

Open **http://127.0.0.1:8000**. The API documentation is at **http://127.0.0.1:8000/docs**.

Select one of the five **Best-performing road clips**. Use **Open tested output** to view its saved verified result immediately, or **Analyze sample** to run your current settings. Use **Change file** to upload a video/image. Once analysis finishes, play the annotated video. Enable **Voice alerts** for readings synchronized to playback; uncertain readings are excluded unless you opt in. You can also click a reading's speaker or timestamp. Voice uses your browser's Speech Synthesis API; Chrome or Safari with an installed English voice works best.

If port 8000 is occupied, use `PORT=8001 ./run_signtrace.sh`. The frontend uses relative API URLs and works on either port.

### Clean installation

The existing `.venv` uses Python 3.15 and does not contain PyTorch. This app uses the prepared `.venv312` environment instead. To recreate it with Python 3.12 installed:

```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r traffic_sign_app/requirements.txt
brew install ffmpeg tesseract
./run_signtrace.sh
```

FFmpeg converts results to H.264/yuv420p with MP4 fast start and preserves source audio. Tesseract is optional for model-only classification, but is required for reliable text/number reading. Both tools were already present on this Mac.

## What was broken

1. The existing Python 3.15 `.venv` had none of the inference libraries installed.
2. Both the input and old annotated output used the MPEG-4 Part 2 codec (`mp4v`), which browser players often cannot decode. The old annotated file also contained only 249 of the input's 270 frames.
3. The model is a cropped-sign classifier, not a full-frame sign detector. The old notebook used only red/blue proposals and discarded every illustrated-sign prediction below 80%, leaving the alert CSV empty.
4. Illustrated signs differ from photographed GTSRB images. The supplied checkpoint's stored best validation score is 82.65%; that is not video detection accuracy. Some illustrated signs yield incorrect or weak model predictions.
5. A track could receive multiple votes in one frame, and reusing cached OCR could incorrectly appear to provide several independent observations.

## Implemented behavior

- Strict checkpoint loading with `weights_only=True`, architecture compatibility, finite parameters, SHA-256 identity and a forward-pass check.
- CPU, CUDA or Apple Silicon MPS inference with the checkpoint's exact preprocessing.
- Video localization uses a published GTSDB SSD MobileNet graph on overlapping tiles, refreshed at the selected frame interval. Your unchanged `.pth` model classifies those sign crops; it is not replaced by the detector. One-to-one spatial matching tracks the boxes. Color/shape proposals remain the image-scan/fallback path; filled-red STOP proposals must independently read STOP through OCR.
- Classification at the selected frame interval. Every crop is sent through your actual `.pth` model.
- Text reading on detected sign crops, including isolated speed digits, STOP and readable rectangular boards. Text readings are labeled **OCR**, not presented as model predictions.
- Distinct observation confirmation. One cached OCR result cannot trigger three confirmation votes. Each tracked label is announced once rather than on every frame.
- Weak predictions remain uncertain. Low-confidence contour guesses cannot become spoken warning events. Blue sky/board proposals cannot be classified as speed-limit signs.
- Bounded background video jobs, progress preview, cancellation, saved history, image scan and cropped-sign classification.
- Full-frame-count processing, H.264 playback, video download and CSV/JSON event export.
- Browser voice alerts follow the processed video's playback clock; speech is not dumped out during accelerated background processing.
- Videos and model inference remain on this computer. The server binds to loopback only.

## Notebook

`../video_sign_alert.ipynb` now controls this same backend and displays browser-compatible output. It can start the backend using `.venv312` even if the notebook client kernel uses a different Python version. It streams uploads, monitors progress, downloads results to `annotated_road.mp4` / `traffic_sign_alerts.csv`, and embeds the H.264 video.

Original notebook/output/log backups are in `notebook_backup/`. The training and dataset notebooks are preserved.

## Files

```text
traffic_sign_app/
  backend/vision.py       checkpoint, region proposals, OCR, tracking and annotations
  backend/jobs.py         background jobs, progress and H.264 export
  backend/main.py         FastAPI routes and frontend hosting
  frontend/index.html    responsive interface
  frontend/style.css
  frontend/app.js         uploads, job progress, playback and speech synchronization
  tests/test_app.py       model, OCR, tracker, API and actual-video regressions
  runtime/               input uploads and generated results, saved only locally
  requirements.txt
```

The root `run_signtrace.sh` starts both layers. The frontend requires no separate Node installation or build.

## API

- `GET /api/health`, `/api/model`
- `POST /api/jobs` — multipart video upload; `threshold`, `stride`, `ocr`
- `POST /api/jobs/demo` — analyze the allowlisted `sample_id` (default `city-crossing`)
- `GET /api/jobs`, `/api/jobs/{id}` — history/progress
- `POST /api/jobs/{id}/cancel`
- `GET /api/jobs/{id}/preview`, `/video`, `/alerts.csv`, `/events.json`
- `POST /api/images` — image scan, with `cropped=true` for a pre-cropped sign

Videos are limited to 250 MB and images to 20 MB. Model checkpoint configuration is server-side; arbitrary paths and weights cannot be uploaded through the frontend.

## Verify

```bash
.venv312/bin/python -m pytest -q traffic_sign_app/tests
```

Tests use the actual supplied checkpoint/video plus isolated fixtures. They verify numeric/text readings, track identity and observation counting, API error handling, H.264 frame count and byte-range delivery, cancellation and audio preservation. They do not claim a new model accuracy score.

Localization and classification can still miss small signs, poor lighting, occlusion or unfamiliar designs. Detector scores and classifier scores are separate. Stored training validation accuracy is not video accuracy. This is a project demonstration.


## Reviewed video library

Thirty candidate clips were screened with the initial pipeline, followed by trained-detector comparisons and independent complete runs of the five final excerpts. The five are **non-overlapping excerpts from two real drives**, selected to show correct visible readings: City crossing, City junction, 30 & direction, Street island, Town speed 30. They are curated examples, not a held-out generalization benchmark. See [benchmark/REPORT.md](benchmark/REPORT.md) for the full comparison, rejected clips, missed-sign limitations and exact source windows.

`GET /api/samples` returns metadata, Range-enabled preview URLs, posters and the saved reviewed job ID for each clip. Switching is disabled while a job runs. Each final clip was verified to produce its stated readings at threshold .55, stride 3, three hits and OCR enabled. Results may differ with other settings or devices. No source-video timing or filename is used to invent predictions.

Source attribution: [samples/SOURCES.md](samples/SOURCES.md). Localization graph provenance: [models/SOURCES.md](models/SOURCES.md). The original clips and comparison outputs remain on disk as evidence; they are no longer offered in the main five-video selector.


## Continuous video with more sign types

The first library option is now **Long drive · 7 sign types**, a complete 4:04 real German drive. The other four options remain short tested clips. Seven timestamp buttons navigate to manually verified physical signs: 30 km/h, priority road, straight-or-right, keep-right, 40 km/h, no-stopping and a 30-zone sign. These references are explicitly separate from model predictions. The latter three types are outside this checkpoint’s supported classes.

Use **Open full analysis** to inspect the actual complete pipeline output, or **Analyze sample** to rerun it. The full run includes missed and wrong readings; 11 emitted events do not mean seven correctly recognized types. See [benchmark/LONG_DRIVE_REVIEW.md](benchmark/LONG_DRIVE_REVIEW.md) and [samples/LONG_DRIVE_SOURCE.md](samples/LONG_DRIVE_SOURCE.md).
