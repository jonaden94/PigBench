# MOTRv2
All paths in this guide are relative to the ``tracking/motrv2`` directory of the repository, which therefore needs to be set as the working directory. Before starting this guide, make sure that the necessary training and inference data preparations have been completed as described [here](../README.md)

## Training
If you are only interested in model inference, you can skip this section. Training MOTRv2 requires model weights pre-trained on DanceTrack as initialization as well as a JSON file containing bounding box priors from Co-DINO:

| Files        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| Bounding Box Priors   | ```python tools/download/download.py --name bbox_priors --root ../data/datasets/PigTrack``` | todo MB          |
| MOTRv2 DanceTrack weights   | ```python tools/download/download.py --name motrv2_dancetrack --root ../data/pretrained/motrv2/full_model``` | todo MB          |

In case that the automatic download does not work, you can manually download the bounding box priors [here](https://doi.org/10.25625/P7VQTP) and the pre-trained MOTRv2 weights [here](https://github.com/megvii-research/MOTRv2?tab=readme-ov-file). The downloaded files should be placed in folders matching the paths specified by the --root argument in the Python commands above.

We also provide a script to compute the bounding box priors. It can be used for new datasets, but for PigTrack and PigDetect this script does not need to be run since the pre-computed bounding boxes were already downloaded above. To run the script, use the following command:
```
python tools/prep/create_bbox_prior.py
```

Training MOTRv2 requires 8 GPUs to achieve the performance levels reported in the paper. Use the following command for training
```
bash tools/train/train.sh
```

We also provide a slurm batch script for distributed training in ``tools/train/`` that works with our system configuration (e.g. folder structure and conda path) and GPU cluster. Before using this script, all paths and cluster-related specifications need to be adapted to match your setup.

## Inference
At inference time, bounding box priors need to be created for the input video. For this, the CO-DINO model pre-trained on pig data is required:

| File        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| Co-DINO pig weights   | ```python tools/download/download.py --name codino_weights --root ../data/pretrained/codino``` | 900 MB          |
| MOTRv2 pig weights   | ```python tools/download/download.py --name motrv2_weights --root ../data/pretrained/motrv2/full_model``` | todo MB          |

In case that the automatic download does not work, you can manually download the Co-DINO weights [here](https://doi.org/10.25625/I6UYE9) and the MOTRv2 weights [here](https://doi.org/10.25625/P7VQTP). The downloaded files should be placed in the folders specified by the --root argument in the Python commands above.

Unlike training, inference can also be run on a single GPU with the following command:
```
bash tools/inference/inference.sh
```

You can run inference on custom videos by modifying the --video_dir arg in ``tools/inference/inference.sh``.

We also provide a slurm batch script for distributed inference (``tools/inference/inference_slurm.sh``) that works with our system configuration (e.g. folder structure and conda path) and GPU cluster. This way, multiple videos can be processed at the same time. Before using this script, all paths and cluster-related specifications need to be adapted to match your setup.

## Evaluation
Once inference has been run on the two test videos as described above, you can evaluate the results by comparing them with the ground truth using the following command:

```
bash tools/evaluation/eval.sh
```

You can evaluate the test results of any method on the PigTrack test set by adjusting the ``tracker_dir`` argument in ``tools/evaluation/eval.sh``. You just need to make sure that the predicted output complies with the format required by the [TrackEval](https://github.com/JonathonLuiten/TrackEval) repository since our evaluation script uses this repository.

## Licensing
This folder combines code from multiple open-source projects with different licenses:

1. mmdetection
   - **Repository:** https://github.com/open-mmlab/mmdetection
   - **License:** Apache-2.0  
   - **Location:** `tracking/motrv2/detector`

2. TrackEval
   - **Repository:** https://github.com/JonathonLuiten/TrackEval
   - **License:** MIT  
   - **Location:** `tracking/motrv2/TrackEval`

3. MOTRv2 
   - **Repository:** https://github.com/megvii-research/MOTRv2
   - **License:** MIT  
   - **Location:** Other files in ``tracking/motrv2``
   - **Note:** The original MOTRv2 code includes portions of code copied from other projects licensed under the Apache-2.0 license (see file headers for details).

If files from Apache-2.0 licensed projects have been copied or modified from their original versions, this is indicated in the file headers.
