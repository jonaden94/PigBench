# Pig detection
All paths in this guide are relative to the ``detection`` directory of the repository, which therefore needs to be set as the working directory.

## Training
If you are only interested in model inference, you can skip this section. For training, the PigDetect benchmark dataset and the COCO-style train/val/test annotation files are required. Furthermore, to reproduce the results from the paper, you need to download the model checkpoints pre-trained on the COCO dataset. The following commands will download all required data:

| Files        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| PigDetect benchmark dataset   | ```python tools/download/download.py --name pigdetect --root data/PigDetect``` | 1200 MB          |
| COCO-style annotation files   | ```python tools/download/download.py --name coco_annotation_files --root data/PigDetect``` | 14 MB          |
| YOLOX COCO weights   | ```python tools/download/download.py --name yolox_pretrained_weights --root data/pretrained_weights/yolox_coco``` | 718 MB          |
| YOLOv8 COCO weights   | ```python tools/download/download.py --name yolov8_pretrained_weights --root data/pretrained_weights/yolov8_coco``` | 570 MB          |
| Co-DINO COCO weights   | ```python tools/download/download.py --name codino_pretrained_weights --root data/pretrained_weights/codino_coco``` | 900 MB          |

Feel free to add further training data or use other train/val splits for your application at hand! In case that the automatic download does not work, you can manually download PigDetect and the COCO-style annotation files [here](https://doi.org/10.25625/I6UYE9). Similarly, the pre-trained model weights can be downloaded at the model zoos of [mmyolo](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov8) and [mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/projects/CO-DETR). The downloaded files should be placed in the folders specified by the --root argument in the Python commands above.

Once you downloaded the dataset, it needs to be restructured so that it can be used for training:

```
python tools/download/restructure.py
```

You can train the model using mmdetection/mmyolo functionality. The training of any of the YOLOX or YOLOv8 models only requires a single GPU. For example, the following command is used to train the YOLOv8-X model:

```
python train_mmyolo.py configs/yolov8/yolov8_x.py --work-dir outputs/yolov8_x
```

The training commands for all further YOLOX and YOLOv8 models can be found under ``tools/train``. You might need to adjust the batch size in case your GPU does not have sufficient memory. This and other adaptations can be made in the respective config files located in ``configs``. We refer to the [mmdetection documentation](https://mmdetection.readthedocs.io/en/dev-3.x/index.html) for further information on configs.

To train Co-DINO, multiple GPUs are required. If 4 GPUs are available on your system, you can run the following command:
```
bash tools/train/codino/train.sh
```

For all models, we also provide slurm batch scripts in ``tools/train/`` that work with our local GPU cluster and folder structure.

## Inference
To perform inference using the trained pig detection models presented in the paper, you first have to download the trained model weights:

| Files        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| YOLOX pig weights   | ```python tools/download/download.py --name yolox_weights --root data/pretrained_weights/yolox_pigs``` | 718 MB          |
| YOLOv8 pig weights   | ```python tools/download/download.py --name yolov8_weights --root data/pretrained_weights/yolov8_pigs``` | 573 MB          |
| Co-DINO pig weights   | ```python tools/download/download.py --name codino_weights --root data/pretrained_weights/codino_pigs``` | 900 MB          |

In case that the automatic download does not work, you can manually download the files [here](https://doi.org/10.25625/I6UYE9). The downloaded files should be placed in the folders specified by the --root argument in the Python commands above.

After the model weights have been downloaded, you can use mmdetection/mmyolo functionality to load the models and use them for inference. For this, we prepared a [demo notebook](tools/inference/inference_demo.ipynb).

## Evaluation
For evaluation, you first need to ensure that the dataset and the coco annotation files are downloaded and the dataset is restructured as described in the training section.
To obtain mAP and AP50 evaluation metrics with a certain model on the test split, you can use the test functionality provided by mmdetection. For example, to obtain test performance of our Co-DINO model trained for pig detection, run the following command:

```
python test.py configs/co_detr/co_dino_swin.py data/pretrained_weights/codino_pigs/codino_swin.pth --work-dir outputs/co_dino
```

The evaluation commands for all further models trained by us can be found under ``tools/eval``

## Licensing
This folder combines code from multiple open-source projects with different licenses:

1. mmdetection
   - **Repository:** https://github.com/open-mmlab/mmdetection
   - **License:** Apache-2.0  

2. mmyolo
   - **Repository:** https://github.com/open-mmlab/mmyolo
   - **License:** GPL 3.0  

If files from Apache-2.0 or GPL 3.0 licensed projects have been copied or modified from their original versions, this is indicated in the file headers.
Independent contributions are licensed under GPL 3.0, as required by the GPL 3.0 copyleft terms.
