# 『Fashion MNIST 데이터셋을 이용한 CNN 이미지 분류』



---



## 파일 구성

| 폴더 & 파일 이름 | 설명 |
|-----------------|------|
| codes | 분류를 위한 CNN 관련 소스 코드 |
| logs & figure | Scratch 훈련/테스트 그래프 출력 화면 및 로그 캡처 |
| cnn_params.pkl | Scratch부터 직접 학습하여 저장한 분류 모델(.pkl) |
| test_acc.npy | Test 정확도 기록 |
| train_acc.npy | Train 정확도 기록 |




## 요구사항

소스 코드를 실행하려면 아래의 소프트웨어가 설치되어 있어야 합니다.



* 파이썬 3.x

* NumPy

* Matplotlib





## 실행 방법 

scratch 폴더에서 실행하세요. (Fashion MNIST 데이터셋 다운 후 학습 시작)

```

$ cd scratch

$ python codes/dataset/mnist.py

$ python codes/cnn/train_deepnet.py

```



---



## 라이선스



이 저장소의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)를 따릅니다.

상업적 목적으로도 자유롭게 이용하실 수 있습니다.

