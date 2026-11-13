# Sign localization graph

`detection.pb` and `detection.pbtxt` were downloaded from the published OpenCV integration:
https://github.com/MaxRitter/OpenCV_TrafficSign/tree/master/models

That project attributes its detector to Álvaro Arcos-García et al., **Evaluation of deep neural networks for traffic sign detection systems**, Neurocomputing 316 (2018), 332–344:
https://github.com/aarcosg/traffic-sign-detection
https://doi.org/10.1016/j.neucom.2018.08.009

This is an SSD MobileNet V1 detector trained on GTSDB, with three coarse categories: prohibitory, mandatory and danger. The application runs the graph using OpenCV DNN on overlapping tiles to preserve distant signs. Detector scores and classification scores are separate. The original classifier graph from that repository is NOT used.

Your `../../gtsrb_best_model.pth` remains the 43-class classifier. Its weights are unchanged. Tesseract reads actual sign crops where legible. Neither detector labels nor video filenames supply speed numbers or class labels to the classifier.

No Python or executable code was downloaded or run from this repository. The `.pb` graph and `.pbtxt` graph configuration are parsed by OpenCV. Refer to the source projects for attribution and any redistribution terms; this record does not grant additional rights.
