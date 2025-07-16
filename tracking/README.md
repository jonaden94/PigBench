# Pig tracking
All paths in this guide are relative to the ``tracking`` directory of the repository, which therefore needs to be set as the working directory.

## Training data preparation
If you are only interested in model inference, you can skip this section. Training MOTRv2 and MOTIP requires downloading the PigDetect and PigTrack datasets.  Run the following commands to download the datasets:

| Files        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| PigDetect benchmark dataset   | ```python tools/download/download.py --name pigdetect --root data/datasets/PigDetect``` | 1200 MB          |
| PigTrack benchmark dataset   | ```python tools/download/download.py --name pigtrack --root data/datasets/PigTrack``` | 20100 MB          |

In case that the automatic download does not work, you can manually download the datasets [here](https://doi.org/10.25625/I6UYE9) and [here](https://doi.org/10.25625/P7VQTP). For PigTrack, you need to download all individual zip files (pigtrackxxxx.zip) as well as split.txt. The downloaded files should be placed in the folders specified by the --root argument in the Python commands above.

Once the datasets are downloaded, they need to be restructured so that they can be used for training:

```
python tools/download/restructure_pigdetect.py
python tools/download/restructure_pigtrack.py
```

## Inference data preparation
To provide example inputs that can later be used for inference, we need to download the PigTrack videos in MP4 format. Run the following download command:

| File        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| PigTrack MP4 videos   | ```python tools/download/download.py --name pigtrack_videos --root data/datasets/PigTrackVideos``` | 1300 MB          |

In case that the automatic download does not work, you can manually download the videos in zip format [here](https://doi.org/10.25625/P7VQTP). The downloaded file should be placed in the folder specified by the --root argument in the Python command above.

To unzip the file and delete all videos except for two example videos from the test set, run the following command:
```
python tools/download/restructure_videos.py
```

## Tracking models
Once the data is prepared, you can use the provided training and inference functionality for the models evaluated in our paper. If you wanna use one of the SORT-based models, follow the [`BoxMOT guide`](boxmot/README.md). Information on how to use the learning-based models can be found in the [`MOTIP guide`](motip/README.md) and [`MOTRv2 guide`](motrv2/README.md).
