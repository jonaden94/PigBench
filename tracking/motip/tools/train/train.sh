############## Train MOTIP with torch.distributed.run functionality
############## If 8 GPUs are available on your system, run the following command:

python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=29400 \
    main.py \
    --mode train \
    --config ./configs/motip.yaml \
    --exp-name train

# python -m torch.distributed.run \
#     --nproc_per_node=8 main.py \
#     --mode train \
#     --config ./configs/motip.yaml \
#     --exp-name train
