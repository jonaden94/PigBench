############## To train any of the YOLOX models, we use the train_mmyolo.py provided by mmyolo.
############## If there is a GPU available on your system, run one of the following commands:

python train_mmyolo.py configs/yolox/yolox_s_rtmdet_hyp.py --work-dir outputs/yolox_s_rtmdet_hyp
python train_mmyolo.py configs/yolox/yolox_m_rtmdet_hyp.py --work-dir outputs/yolox_m_rtmdet_hyp
python train_mmyolo.py configs/yolox/yolox_l.py --work-dir outputs/yolox_l
python train_mmyolo.py configs/yolox/yolox_x.py --work-dir outputs/yolox_x