# BoxMOT
All paths in this guide are relative to the ``tracking/boxmot`` directory of the repository, which therefore needs to be set as the working directory. Before starting this guide, make sure that the test videos for inference have been downloaded as described [here](../README.md)

## Inference
Running inference on new videos with models from BoxMOT requires the CO-DINO model pre-trained on pig data:
| File        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| Co-DINO pig weights   | ```python tools/download/download.py --name codino_weights --root ../data/pretrained/codino``` | 900 MB          |

In case that the automatic download does not work, you can manually download the Co-DINO weights [here](https://doi.org/10.25625/I6UYE9). The downloaded file should be placed in the folder specified by the --root argument in the Python command above.

To use the BoT-SORT model for inference, run the following command:
```
python main.py --config configs/botsort.yaml
```

Inference for the other models can be run by using the corresponding config file. Running inference on custom videos can be done by modifying the seq_dir arg in ``configs/base.yaml``. You can also use a faster detection model for inference. You can do so by changing the inference_detector_config and inference_detector_checkpoint args in ``configs/base.yaml`` to match one of the models presented in the detection part of this repository (see [here](../../detection/README.md)).

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
   - **Location:** `tracking/boxmot/detector`

2. TrackEval
   - **Repository:** https://github.com/JonathonLuiten/TrackEval
   - **License:** MIT  
   - **Location:** `tracking/boxmot/TrackEval`

3. BoxMOT 
   - **Repository:** https://github.com/mikel-brostrom/boxmot
   - **License:** AGPL  
   - **Location:** ``tracking/boxmot/boxmot``

If files from Apache-2.0 licensed projects have been copied or modified from their original versions, this is indicated in the file headers.
