####################################### training with torch.distributed.run (only works on one node for me and only tested for MOTRv2 so far)
python -m torch.distributed.run \
    --nproc_per_node=8 main.py \
    --mode train \
    --config configs/motrv2.yaml \
    --exp_name train
