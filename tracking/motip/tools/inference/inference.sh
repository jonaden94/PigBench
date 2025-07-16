
python main.py \
    --mode video_inference \
    --config ./configs/motip.yaml \
    --exp-name inference \
    --inference-model ../data/pretrained/motip/full_model/MOTIP.pth \
    --video-dir ../data/datasets/PigTrackVideos \
    --visualize-inference True \

####################################### inference with torch.distributed.run (only works on one node for me) 
# python -m torch.distributed.run \
#     --nproc_per_node=4 main.py \
#     --mode video_inference \
#     --config ./configs/motip.yaml \
#     --exp-name inference \
#     --inference-model ../data/pretrained/motip/full_model/MOTIP.pth \
#     --video-dir ../data/datasets/PigTrackVideos \
#     --visualize-inference True