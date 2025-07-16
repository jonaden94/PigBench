python tools/evaluation/eval.py \
    --gt_dir ../data/datasets/PigTrack/test \
    --tracker_dir outputs/inference/video_inference/PigTrackVideos/MOTRv2/results/tracker \
    --seqmap_path ../data/datasets/PigTrack/test_seqmap_2videos_only.txt # evalution on the two test videos from the inference demo
    # --seqmap_path ../data/datasets/PigTrack/test_seqmap.txt # use this when evaluating on the entire test set