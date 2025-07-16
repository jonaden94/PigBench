############## Train Co-dino with torch.distributed.launch functionality
############## If 4 GPUs are available on your system, run the following command:

python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=4 \
    --master_port=29400 \
    train_mmdet.py \
    configs/co_detr/co_dino_swin.py \
    --work-dir outputs/co_dino \
    --launcher pytorch