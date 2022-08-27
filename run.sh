#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=13584
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=3
export OMP_NUM_THREADS=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

python3 -m torch.distributed.run ${DISTRIBUTED_ARGS} /data/disk2/private/roufaen/loss_truncation/train.py
