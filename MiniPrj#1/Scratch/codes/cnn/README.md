## 파일 설명
| 파일명 | 파일 용도 |
|:--   |:--      |
| deep_convnet_my.py | CNN을 구현한 소스 + common/layers.py의 BatchNormalization 활용 |
| cnn_params.pkl | Scratch부터 직접 학습하여 저장한 분류 모델(.pkl) |
| train_deepnet.py | deep_convnet_my.py의 신경망을 학습시킵니다. 몇 시간은 걸리기 때문에 다른 코드에서는 미리 학습된 가중치인 cnn_params.pkl을 읽어서 사용하는 것이 가능합니다. |
