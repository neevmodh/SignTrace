# Real road sample

`../real_road.mp4` contains actual moving-camera street footage, not animation. It was downloaded on 25 October 2026 from the public test asset in Suprit-U/Real-Time-Traffic-Sign:

- Source file: https://github.com/Suprit-U/Real-Time-Traffic-Sign/blob/main/Test/vidd1.mp4
- Download: https://raw.githubusercontent.com/Suprit-U/Real-Time-Traffic-Sign/main/Test/vidd1.mp4
- Repository: https://github.com/Suprit-U/Real-Time-Traffic-Sign

The complete 17.733-second clip (532 frames at 30 fps) was resized from 1280×720 to 960×540 and encoded as H.264/yuv420p with AAC audio and faststart for browser playback. No sign annotations or generated objects were added to the input. Original downloaded file Git blob SHA: d93c7eb486dbaf72658554962b15318d7efd8abe.

The source repository does not display an explicit media license. This local sample is for testing; verify rights with the source owner before redistributing or publishing the footage. Repository attribution does not establish original authorship of the footage.

The application still uses your `gtsrb_best_model.pth`; no weights or predictions from the source repository are used. `road.mp4` remains the previous illustrated fixture. Results on this real clip are predictions, not ground-truth annotations.
