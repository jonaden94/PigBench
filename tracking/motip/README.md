# MOTIP
All paths in this guide are relative to the ``tracking/motip`` directory of the repository, which therefore needs to be set as the working directory. Before starting this guide, make sure that the necessary training and inference data preparations have been completed as described [here](../README.md)

## Training

If you are only interested in model inference, you can skip this section. Training MOTIP requires pre-trained DETR weights that can be used as initialization. Run the following commands to download the weights:

| Files        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| DETR COCO weights   | ```python tools/download/download.py --name detr_pretrained_coco --root ../data/pretrained/motip/detr``` | 467 MB          |
| DETR pig weights   | ```python tools/download/download.py --name detr_pretrained_pigs --root ../data/pretrained/motip/detr``` | 156 MB          |

In case that the automatic download does not work, you can manually download the DETR weights pre-trained on COCO [here](https://github.com/MCG-NJU/MOTIP/blob/main/docs/GET_STARTED.md) and pre-trained on pig data [here](https://doi.org/10.25625/P7VQTP). The downloaded files should be placed in the folders specified by the --root argument in the Python commands above.

After downloading, you can first fine-tune the DETR weights pre-trained on COCO using pig data, as described in the paper. Alternatively, you can also skip this step and directly use the DETR weights pre-trained on pig data (also downloaded above) to train the full model. It should be noted that training MOTIP requires 8 GPUs to achieve the performance levels reported in the paper. If 8 GPUs are available, use the following command for pre-training: 
```
bash tools/train/pretrain.sh
```

To train the full model, run the following command:
```
bash tools/train/train.sh
```

We also provide slurm batch scripts for distributed training in ``tools/train/`` that work with our system configuration (e.g. folder structure and conda path) and GPU cluster. Before using these scripts, all paths and cluster-related specifications need to be adapted to match your setup.

## Inference

Model inference requires the pre-trained MOTIP weights for pig tracking. Run the following command to download the weights:

| Files        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| MOTIP pig weights   | ```python tools/download/download.py --name motip_weights --root ../data/pretrained/motip/full_model``` | 227 MB          |

Unlike training, inference can also be run on a single GPU with the following command:
```
bash tools/inference/inference.sh
```

You can run inference on custom videos by modifying the --video-dir arg in ``tools/inference/inference.sh``. 

We also provide a slurm batch script for distributed inference (``tools/inference/inference_slurm.sh``) that works with our system configuration (e.g. folder structure and conda path) and GPU cluster. This way, multiple videos can be processed at the same time. Before using this script, all paths and cluster-related specifications need to be adapted to match your setup.

## Evaluation
Once inference has been run on the two test videos as described above, you can evaluate the results by comparing them with the ground truth using the following command:

```
bash tools/evaluation/eval.sh
```

You can evaluate the test results of any method on the PigTrack test set by adjusting the ``tracker_dir`` argument in ``tools/evaluation/eval.sh``. You just need to make sure that the predicted output complies with the format required by the [TrackEval](https://github.com/JonathonLuiten/TrackEval) repository since our evaluation script uses this repository.

## Licensing
This folder combines code from multiple open-source projects with different licenses:

1. TrackEval
   - **Repository:** https://github.com/JonathonLuiten/TrackEval
   - **License:** MIT  
   - **Location:** `tracking/motip/TrackEval`

2. MOTIP 
   - **Repository:** https://github.com/MCG-NJU/MOTIP
   - **License:** Apache-2.0  
   - **Location:** Other files in ``tracking/motip``
   - **Note:** The original MOTIP code includes portions of code copied from other projects licensed under the Apache-2.0 license (see file headers for details).

If files from Apache-2.0 licensed projects have been copied or modified from their original versions, this is indicated in the file headers.
