#!/bin/bash

# 현재 스크립트의 파일 이름을 가져옵니다 (확장자 제외)
SCRIPT_NAME=$(basename "$0" .sh)

# nohup 명령을 실행하고 백그라운드로 보냅니다
nohup python -u ../sheeprl.py exp=mudreamer \
                     env=unity_env \
                     env.num_envs=4 \
                     fabric.accelerator=cuda \
                     algo.total_steps=100000 \
                     algo.per_rank_batch_size=16 \
                     checkpoint.keep_last=null \
                     checkpoint.every=512 \
                     metric.log_every=100 \
                     buffer.from_numpy=True \
                     buffer.size=100000 \
                     env.wrapper.file_name=/home/calm03/data/project/calmlab_rl_model/unity_build_env/Sample.x86_64 \
                     > "${SCRIPT_NAME}.log" 2>&1 &

# 방금 시작한 백그라운드 프로세스의 PID를 가져옵니다
PID=$!

# PID를 출력합니다
echo "프로세스가 시작되었습니다. PID: $PID"