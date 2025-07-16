############## Evaluation commands:

# YOLOv8
python test.py configs/yolov8/yolov8_s.py data/pretrained_weights/yolov8_pigs/yolov8_s.pth --work-dir outputs/yolov8_s
python test.py configs/yolov8/yolov8_m.py data/pretrained_weights/yolov8_pigs/yolov8_m.pth --work-dir outputs/yolov8_m
python test.py configs/yolov8/yolov8_l.py data/pretrained_weights/yolov8_pigs/yolov8_l.pth --work-dir outputs/yolov8_l
python test.py configs/yolov8/yolov8_x.py data/pretrained_weights/yolov8_pigs/yolov8_x.pth --work-dir outputs/yolov8_x

# YOLOX
python test.py configs/yolox/yolox_s_rtmdet_hyp.py data/pretrained_weights/yolox_pigs/yolox_s.pth --work-dir outputs/yolox_s
python test.py configs/yolox/yolox_m_rtmdet_hyp.py data/pretrained_weights/yolox_pigs/yolox_m.pth --work-dir outputs/yolox_m
python test.py configs/yolox/yolox_l.py data/pretrained_weights/yolox_pigs/yolox_l.pth --work-dir outputs/yolox_l
python test.py configs/yolox/yolox_x.py data/pretrained_weights/yolox_pigs/yolox_x.pth --work-dir outputs/yolox_x

# Co-DINO
python test.py configs/co_detr/co_dino_swin.py data/pretrained_weights/codino_pigs/codino_swin.pth --work-dir outputs/co_dino