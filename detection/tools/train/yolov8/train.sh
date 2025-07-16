############## To train any of the YOLOv8 models, we use the train_mmyolo.py provided by mmyolo.
############## If there is a GPU available on your system, run one of the following commands:

python train_mmyolo.py configs/yolov8/yolov8_s.py --work-dir outputs/yolov8_s
python train_mmyolo.py configs/yolov8/yolov8_m.py --work-dir outputs/yolov8_m
python train_mmyolo.py configs/yolov8/yolov8_l.py --work-dir outputs/yolov8_l
python train_mmyolo.py configs/yolov8/yolov8_x.py --work-dir outputs/yolov8_x