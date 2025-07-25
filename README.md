# Benchmarking pig detection and tracking under diverse and challenging conditions
![pigdetect_example_repo_resized](https://github.com/user-attachments/assets/19fde59e-e786-4593-8171-15c3709bdebc)
![pigtrack_smaller](https://github.com/user-attachments/assets/16086a4c-75b5-496d-bbc9-b9734adb2277)

To ensure animal welfare and effective management in pig farming, monitoring individual behavior is a crucial prerequisite. While monitoring tasks have traditionally been carried out manually, advances in machine learning have made it possible to collect individualized information in an increasingly automated way. Central to these methods is the localization of animals across space (object detection) and time (multi-object tracking). Despite extensive research of these two tasks in pig farming, a systematic benchmarking study has not yet been conducted. In this work, we address this gap by curating two datasets: PigDetect for object detection and PigTrack for multi-object tracking. The datasets are based on diverse image and video material from realistic barn conditions, and include challenging scenarios such as occlusions or bad visibility. For object detection, we show that challenging training images improve detection performance beyond what is achievable with randomly sampled images alone. Comparing different approaches, we found that state-of-the-art models offer substantial improvements in detection quality over real-time alternatives. For multi-object tracking, we observed that SORT-based methods achieve superior detection performance compared to end-to-end trainable models. However, end-to-end models show better association performance, suggesting they could become strong alternatives in the future. We also investigate characteristic failure cases of end-to-end models, providing guidance for future improvements. The detection and tracking models trained on our datasets perform well in unseen pens, suggesting good generalization capabilities. This highlights the importance of high-quality training data. The datasets and research code are made publicly available to facilitate reproducibility, re-use and further development.

The preprint of the paper is available [here](https://arxiv.org/abs/2507.16639). The datasets and pre-trained model weights associated with this work are available [here](https://doi.org/10.25625/I6UYE9) and [here](https://doi.org/10.25625/P7VQTP). This repository also includes automatic download commands to obtain all necessary files for training and inference. So you do not need to download them manually. Simply follow the instructions in this README.

## Setup and requirements

Import information:
- The setup has been tested on a linux machine. We cannot provide any information for other operating systems. 
- Your GPUs and NVIDIA driver must be compatible with the CUDA version specified in our setup (version 11.8). We cannot provide any information for environment setup with other CUDA versions.
- MOTIP and MOTRv2 use custom CUDA operators, which require a suitable GCC compiler to build them (e.g. version 11.4). Furthermore, the compilation relies on several CUDA-related environment variables that might not be automatically set by your system. Therefore, this part of the environment setup is error prone, but it is also not required if you do not plan to use these models.
- The setup script might throw some warnings and potentially also an error that there are some incompatibilities with the "requests" package. This can be ignored.

To set up the environment, we recommend using Conda. If Conda is installed and activated, run either of the following two setup scripts depending on which models you plan to use. If you plan to only use the detection models and SORT-based tracking models, run the following script:
```
source _setup/setup.sh
```

If you want to create an environment that supports all models in this repository (including MOTIP and MOTRv2), run the following script instead:
```
source _setup/setup_with_e2e_models.sh
```
## Repository overview

This repository provides functionality for training, evaluation and inference of pig detection and tracking models. Information on how to use the detection functionality can be found in the [detection guide](detection/README.md). Information on tracking can be found in the [tracking guide](tracking/README.md). All models in this repository require GPU access for training. While inference might also work on a CPU (we did not test this for all models though), it is much slower than on a GPU. Therefore, we highly recommend using a GPU for inference as well. 

## Licensing

This repository is a collection of several independent code bases. Please refer to the LICENSE file within each subdirectory for the specific licensing terms:

- `detection/` – GNU GPL v3 
- `tracking/boxmot/` – GNU AGPL v3  
- `tracking/motip/` – Apache-2.0
- `tracking/motrv2/` – Apache-2.0

Any code outside those subdirectories is licensed under the MIT license.

## Acknowledgements

This work was funded with NextGenerationEU funds from the European Union by the Federal Ministry of Research, Technology and Space under the funding code 16DKWN038. The responsibility for the content of this publication lies with the authors.

![combined](https://github.com/user-attachments/assets/20af25da-011a-4382-8077-8f3237dccf58)

This repository builds on several existing object detection and multi-object tracking code bases:

| Project | Codebase |
|--------|----------|
| [MOTRv2](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_MOTRv2_Bootstrapping_End-to-End_Multi-Object_Tracking_by_Pretrained_Object_Detectors_CVPR_2023_paper.pdf) | [github.com/megvii-research/MOTRv2](https://github.com/megvii-research/MOTRv2) |
| [MOTIP](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf) | [github.com/MCG-NJU/MOTIP](https://github.com/MCG-NJU/MOTIP) |
| BoxMOT | [github.com/mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot) |
| [MMDetection](https://arxiv.org/abs/1906.07155) | [github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) |
| MMYOLO | [github.com/open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) |


These code bases are in turn built on code from many previous works including [TrackEval](https://github.com/JonathonLuiten/TrackEval), [MOTR](https://github.com/megvii-research/MOTR), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [ByteTrack](https://github.com/FoundationVision/ByteTrack), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [OC-SORT](https://github.com/noahcao/OC_SORT), [DanceTrack](https://github.com/DanceTrack/DanceTrack) and [BDD100K](https://github.com/bdd100k/bdd100k). We thank the authors of all of these projects for making their work publicly available.
