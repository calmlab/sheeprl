# Unity env 세팅 및 사용법

이 문서에서는 빌드된 unity 파일을 세팅하는 방법과 사용하는 방법에 대해서 기술합니다. 

### 1. requirements 설치 

unity 환경을 env로 wrapping하기 위한 라이브러리인 mlagent-envs는 따로 requirements.txt에 기재되어있습니다.

```bash
pip install -r requirements.txt
```

### 2. unity환경으로 학습시키기.

학습 및 eval을 실행시키는 sh파일을 모아놓은 scripts에 unity환경에서 트레이닝 할수 있는 샘플코드가 마련되어 있습니다.

```bash
cd scripts
sh mudreamer_unity_env.sh
```

### 3. TODO

    * 아직 학습 진행중에 기록되는 동영상이 제대로 기록되지 않아서 확인중에 있습니다. 
        * 대신 수동으로 sheeprl/notebooks/mudreamer_imagination.ipynb에서 checkpoint_path를 지정하고, 노트북 전체 script를 실행하면 해당 체크포인트의 에이전트가 환경과 상호작용한 visual결과를 gif로 받아 볼수 있습니다.
    * observation_space에 cnn을 사용하는 visual observation이외에 다른 observation을 사용할수 있는 기능은 구현되어 있지 않습니다. 
    * action_space의 유형에 따라서 config가 어떻게 설정되어져야 하는지 확인이 필요합니다. 