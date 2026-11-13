# Traffic-sign project: models, training and processing components

Verified against `train_model.ipynb`, the actual `gtsrb_best_model.pth` checkpoint, and `traffic_sign_app/backend/`. The checkpoint verifies architecture, class count, preprocessing and stored validation result. Loss, optimizer and training schedule below come from the current training notebook; optimizer/loss settings are not saved inside the checkpoint.

## Models and recognition tools

| Component | Actually used | Role |
|---|---|---|
| Sign classifier | MobileNetV3-Large | Classifies cropped signs into 43 GTSRB classes using `gtsrb_best_model.pth`. |
| Classifier initialization during training | ImageNet-pretrained torchvision weights | Transfer learning; the output layer is replaced with 43 outputs. Runtime loads the user's saved weights without downloading ImageNet weights. |
| Classifier parameters | 4,257,115 | Counted from the architecture loaded with the checkpoint. |
| Classifier head | Linear 960→1280, Hardswish, Dropout 0.2, Linear 1280→43 | Actual active MobileNetV3 head. |
| Video sign detector | Pretrained SSD MobileNet V1, trained on GTSDB | Locates sign boxes on overlapping image tiles. Loaded as `detection.pb` + `detection.pbtxt` through OpenCV DNN. Three broad detection groups: prohibitory, mandatory and danger. |
| Text reader | Tesseract through pytesseract | Reads speed digits, STOP and supported readable text crops. Separate from neural classifier predictions. |
| Optional classifier | ResNet18 | Implemented as an alternative in the notebook; not the active saved model. Its custom head uses 512→256→43, ReLU and Dropout 0.30. |

Detector provenance: [MaxRitter/OpenCV_TrafficSign](https://github.com/MaxRitter/OpenCV_TrafficSign), with the original detector attributed to [aarcosg/traffic-sign-detection](https://github.com/aarcosg/traffic-sign-detection). See `traffic_sign_app/models/SOURCES.md`.

## Loss and optimization

| Item | Notebook configuration |
|---|---|
| Classification loss | `nn.CrossEntropyLoss(label_smoothing=0.05)` on raw logits. |
| Optimizer | AdamW. |
| Weight decay | 0.0001. |
| Stage 1 | Train classifier head, freeze backbone; 3 configured epochs; learning rate 0.001. |
| Stage 2 | Fine-tune the final 3 MobileNet feature blocks and classifier; up to 10 configured epochs. |
| Stage 2 learning rates | Backbone: 0.00001; classifier head: 0.0001. |
| Scheduler | CosineAnnealingLR in both stages, with T_max equal to the configured stage epoch count. |
| Early stopping | Stop after 4 consecutive stage-2 epochs without improved validation accuracy. |
| Gradient clipping | Maximum gradient norm 2.0. |
| Batch size | 32. |
| Random seed | 42. |
| Frozen BatchNorm | Frozen backbone BatchNorm statistics kept in evaluation mode. |
| Best checkpoint | Selected by validation accuracy. Stored checkpoint: stage 2, epoch 9; validation accuracy 82.65%. This is not video recognition accuracy. |

The detector is supplied pretrained and is not trained by this project. Its original training-loss configuration is not included in the supplied inference graph/config, so an exact detector training loss is not asserted here. Tesseract is also pretrained; this app does not train it. No extra training loss is computed during video inference.

## Kernels and neural-network layers

Counts below are from the active MobileNetV3-Large architecture, not the optional ResNet18 or the SSD detector.

| Item | Actual use |
|---|---|
| 1×1 convolution kernels | 46 Conv2d layers: channel projection/expansion and squeeze-excitation operations. |
| 3×3 convolution kernels | 10 Conv2d layers. |
| 5×5 convolution kernels | 6 Conv2d layers. |
| Total convolution layers | 62. |
| Depthwise convolutions | 15 of those convolution layers; these overlap the counts above, rather than being additional layers. |
| Activations | ReLU, Hardswish and Hardsigmoid. |
| Normalization layers | BatchNorm2d. |
| Pooling | Adaptive average pooling, including squeeze-excitation pooling and final feature pooling. |
| Regularization | Dropout probability 0.20 in the active classifier head; label smoothing and AdamW weight decay during training. |
| Output probabilities | Softmax at inference. Training passes logits directly to CrossEntropyLoss. |
| Classical image-processing kernel | 3×3 all-ones morphology kernel for closing HSV color masks in the fallback/image path and STOP candidate support. This is not a learned CNN kernel. |

## Data and image preprocessing

- Dataset: GTSRB, 43 classes; the official test set is separate.
- Training/validation split: approximately 80%/20% of sign tracks within each class, using grouped tracking sequences to reduce leakage.
- Classifier input: RGB, resized to 224×224, converted to tensor.
- Normalization mean: (0.485, 0.456, 0.406); standard deviation: (0.229, 0.224, 0.225).
- Training augmentation: brightness ±20%, contrast ±20%, saturation ±15%, with ColorJitter applied at probability 0.70; rotation ±8°; affine translation up to 6%; scale 0.90–1.10.
- Evaluation: resize, tensor conversion and normalization only.
- Detector input: each full-frame/tile view resized to 300×300 through OpenCV `blobFromImage`, scale factor 1.0 and BGR→RGB channel swap.
- OCR preprocessing: crop enlargement, grayscale/threshold variants, digit isolation and numeric character whitelisting where applicable.

## Video algorithms and settings

| Component | Current implementation/default |
|---|---|
| Object localization | SSD inference on full image and overlapping tiles; detector confidence threshold 0.80. |
| Duplicate-box suppression | OpenCV non-maximum suppression with IoU threshold 0.35; the NMS score gate is 0.35 to also admit independently OCR-confirmed STOP proposals. |
| STOP support | Compact filled-red color/shape proposal accepted only after its pixels OCR as STOP, because the detector's three groups omit STOP. |
| Fallback/image localization | HSV red/blue/green masks, morphology closing, contours, polygon/circularity tests and overlap filtering. |
| Tracking | Custom greedy one-to-one intersection-over-union matching; no separately trained tracking model. |
| Temporal confirmation | Three distinct analyzed observations; cached OCR cannot count as multiple new observations. |
| Classification interval | Every 3 frames by default; configurable. All frames are decoded and written. |
| Classifier confidence threshold | 0.55 by default. Geometry/color constraints can reject incompatible classes. |
| OCR refresh | Approximately every 0.45 seconds per visible track. |
| OCR text segmentation | Tesseract PSM 8 for isolated digits and PSM 6 for general crop text. |
| Browser video encoding | FFmpeg, H.264/libx264, yuv420p, faststart; source audio retained when available by the result encoder. Prepared sample files may omit audio. |
| Speech | Browser Speech Synthesis API, synchronized to result playback. |

## Evaluation metrics

The training notebook computes accuracy, macro precision, macro recall, macro F1, a confusion matrix and a per-class classification report. It also plots training/validation cross-entropy and accuracy. No new video-wide mAP, precision or recall is claimed from the curated demos.

## Software and Python/Jupyter kernels

| Area | Used |
|---|---|
| Training/inference | Python, PyTorch, torchvision, NumPy, Pillow. |
| Image/video processing | OpenCV, FFmpeg. |
| Text reading | Tesseract, pytesseract. |
| Backend | FastAPI, Uvicorn, multipart uploads. |
| Frontend | HTML, CSS, JavaScript; native HTML video and browser speech. |
| Notebook visualizations/evaluation | Matplotlib, scikit-learn, tqdm. |
| Testing | pytest, HTTPX/FastAPI TestClient. |
| Training notebook kernel metadata | `Python (ml)`, kernel name `ml`; recorded Python version 3.13.13. This metadata does not establish the currently selected live Jupyter kernel. |
| Video notebook kernel metadata | `Python 3`, kernel name `python3`; recorded language version 3.12. |
| App backend environment | `.venv312`, verified Python 3.12.14. |
| Hardware selection | Classifier supports CPU, CUDA or Apple MPS; detector uses OpenCV DNN CPU. |

## Main source files

- `train_model.ipynb`: training architecture, loss, optimizer, scheduler and evaluation.
- `gtsrb_best_model.pth`: actual classifier weights and metadata.
- `traffic_sign_app/backend/vision.py`: classifier loading, preprocessing, OCR, fallback proposals, tracking and labels.
- `traffic_sign_app/backend/localizer.py`: SSD sign detection and NMS.
- `traffic_sign_app/backend/jobs.py`: background video processing and encoding.
- `traffic_sign_app/backend/main.py`: web API.
