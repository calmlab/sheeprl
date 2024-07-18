#!/bin/bash
nohup python -u sheeprl.py exp=mudreamer \
                     env=gym \
                     env.id=MsPacman-v4 \
                     env.num_envs=8 \
                     fabric.accelerator=cuda \
                     algo.total_steps=100000 \
                     algo.per_rank_batch_size=16 \
                     checkpoint.keep_last=null \
                     checkpoint.every=512 \
                     metric.log_every=100 \
                     buffer.from_numpy=True \
                     buffer.size=100000 \
                     > Mudremaer.log 2>&1 &