## Deep Learning with Kread - Antonio Gulli & Sujit Pal

### Keras API
케라스는 
1. 모듈 형태, 
2. 최소한의 요소, 
3. 쉽게 확장 가능한 구조 를 갖고 있다.

케라스를 개발한 프랑수와 숄레(Francois Chollet)曰 
"이 라이브러리는 빠른 실험을 가능하게 하는 데 중점을 두고 개발했다. 아이디어에서 결과를 확인하는 데 지연을 최소화하는 것이 좋은 연구를 하는 핵심이다."

케라스는 tensorFlow, Theano 위에서 실행하는 고수준 신경망을 정의한다.
- 모듈성: 모델은 신경망 구축을 위한 독립적인 모듈들의 시퀀스 혹은 그래프다.
- 미니멀리즘: 라이브러리는 파이썬으로 구현되고 각각의 모듈은 간단하고 바로 이해된다(?)
- 쉬운 확장성: 새로운 기능을 갖게 확장할 수 있다.

#### 케라스의 구조 이해
- 텐서: 다차원 배열/행렬
- 케라스 모델 구성
  - 순차 구성(Sequential): 사전에 정의된 다양한 모델을 스택이나 큐처럼 계층을 선형 파이프라인으로 쌓는다.
  - 함수 구성(Functional): 함수 API를 사용해 방향성 비순환 그래프(Directed Acyclic Graph; DAG), 공유 계층이 있는 모델, 다중 출력 모델처럼 복잡한 모델을 정의한다.

#### 사전 정의 신경망 개요
케라스에는 사전에 정의된 많은 신경망 계층이 있다.
- Dense: 완전 연결 신경망 계층
- Sequencial(기본, LSTM, GRU): 순서를 갖는 입력(텍스트, 음석, 시계열, 순서)을 처리하는 신경망의 한 종류. 이전에 나왔던 요소에 영향을 받음.
- CNN Layer & Pooling Layer: 합성곱 신경망은 합성곱과 풀링 연산을 사용해 추상화 단계를 올려가면서(=점진적 추상화) 점점 더 복잡한 모델을 학습. 인간의 두뇌의 시각 모델과 유사.
- 일반화: 과적합(overfitting)을 방지하는 방법.
  - kernel_regularizer: 가중치 행렬에 적용되는 일반화 함수
  - bias_regularizer: 바이어스 벡터에 적용되는 일반화 함수
  - activity_regularizer: 계층의 출력에 적용되는 일반화 함수(활성화)
  - 일반화를 위해 Dropout을 일반적으로 사용한다.
    - rate: 0~1 사이의 실수. 드롭아웃을 적용할 입력 유닛의 비율
    - noise_shape: 입력과 곱할 이진 드롭아웃 마스크의 형태를 나타내는 1D 정수 텐서
    - seed: 랜덤 시드로 사용할 정수
    
- 배치 정규화: 학습 속도를 더 빠르게 하고 일반적으로 더 나은 정확도를 달성하는 방법
Activation function
- Sigmoid
- Linear
- tanh (hyperbolic tangent)
- ReLU (Rectified Linear Unit)
Loss Function (=Cost Function)
- Accuracy: for a Classification problem
  - binary_accuracy: 이진 분류 문제의 모든 예측에서 평균 정확도 비율
  - categorical_accuracy: 다범주 분류 문제의 모든 예측에서 평균 정확도 비율
  - sparse_categorical_accuracy: 희소한 대상에 유용함
  - top_k_categorical_accuracy: top_k 안에 목적 범주가 포함됐을 때 성공
- Error: 예측한 값과 실제 관찰한 값 사이의 차이
  - MSE(Mean Squared Error): 예측한 값과 목표 값 사이의 평균 제곱 오차
  - RMSE(Root Mean Squared Error): 예측한 값과 목표 값 사이의 평균 제곱근 오차
  - MAE(Mean Absolute Error): 예측한 값과 목표 값 사이의 평균 절댓값 오차
  - MAPE(Mean Absolute Percentage Error): 예측한 값과 목표 값 사이의 평균 오차 백분율 값
  - MSLE(Mean Squared Log Error): 예측한 값과 목표 값 사이의 평균 제곱 로그 오차
- hinge loss: 일반적으로 분류기를 학습하는데 사용
  - hinge: max(1-ytrue x ypred, 0)
  - squared hinge: hinge^2
- categorical hinge
  - using categorical_crossentropy
  - using binary_crossentropy
metrics: 메트릭 함수는 목적 함수(objective function)와 유사하다. 유일한 차이점은 모델을 학습할 때 메트릭을 평가한 결과를 사용하지 않는다는 점이다.
optimizer: SGD, RMSprop, Adam   

