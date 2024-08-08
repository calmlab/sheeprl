# Unity env 세팅 및 사용법

이 문서에서는 빌드된 unity 파일을 세팅하는 방법과 사용하는 방법에 대해서 기술합니다. 

### 1. requirements 설치 

unity 환경을 env로 wrapping하기 위한 라이브러리인 mlagent-envs는 따로 requirements.txt에 기재되어있습니다.

```bash
pip install -r requirements.txt
```

설치 이후에 `/site-packages/mlagents_envs/envs/unity_gym_env.py` 파일의 import를 수정해주어야 합니다. 
    * `requirements.txt`에서 설치한 `mlagents-envs`과 `sheeprl`에서 사용하는 `gym`을 같은 `gym`을 import 하도록 맞춰주는 작업입니다.

```python
import itertools

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym # <---- 이곳
from gymnasium import error, spaces # <----- 이곳

from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs import logging_util
```

### 2. 빌드된 unity 환경 세팅하기.

빌드된 unity 환경은 `sheeprl/unity_env`에 zip파일로 압축되어 있습니다. 

압축을 푼 폴더에 `---.x86_64`의 파일경로를 실행파일 script의 옵션으로 지정해주어야 합니다. 

`scripts/mudreamer_unity_env.sh`에 다음과 같이 지정하면 됩니다. 


```sh
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
                     env.wrapper.file_name={이곳에 ---.x86_64파일의 경로를 기재하면 됩니다.} \
                     > "${SCRIPT_NAME}.log" 2>&1 &

# 방금 시작한 백그라운드 프로세스의 PID를 가져옵니다
PID=$!

# PID를 출력합니다
echo "프로세스가 시작되었습니다. PID: $PID"
```

이후에 유니티 실행 파일의 권한을 수정하여야 합니다. 터미널에 다음 명령어를 입력하세요.
```bash
chmod -R 755 {.x86_64 경로}
```

### 3. unity환경으로 학습시키기.

이후에 수정된 sh 파일을 실행시키면 됩니다.

```bash
cd scripts
sh mudreamer_unity_env.sh
```

### 3. TODO

    * 아직 학습 진행중에 기록되는 동영상이 제대로 기록되지 않아서 확인중에 있습니다. 
        * 대신 수동으로 sheeprl/notebooks/mudreamer_imagination.ipynb에서 checkpoint_path를 지정하고, 노트북 전체 script를 실행하면 해당 체크포인트의 에이전트가 환경과 상호작용한 visual결과를 gif로 받아 볼수 있습니다.
    * observation_space에 cnn을 사용하는 visual observation이외에 다른 observation을 사용할수 있는 기능은 구현되어 있지 않습니다. 
    * action_space의 유형에 따라서 config가 어떻게 설정되어져야 하는지 확인이 필요합니다. 