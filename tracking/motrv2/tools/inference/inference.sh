python main.py \
    --mode video_inference \
    --config configs/motrv2.yaml \
    --exp_name inference \
    --inference_model ../data/pretrained/motrv2/full_model/MOTRv2.pth \
    --video_dir ../data/datasets/PigTrackVideos \
    --visualize_inference True

####################################### inference with torch.distributed.run (only works on one node for me) 
# python -m torch.distributed.run \
#     --nproc_per_node=4 main.py \
#     --mode video_inference \
#     --config configs/motrv2.yaml \
#     --exp_name inference \
#     --inference_model ../data/pretrained/motrv2/full_model/MOTRv2.pth \
#     --video-dir ../data/datasets/PigTrackVideos \
#     --visualize_inference True