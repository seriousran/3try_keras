reference: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1

## Learning Rate Schedules
Learning rate(학습 속도) Schedule은 미리 정의된 Schedule 방식에 따라 training(학습)중에 (Learning rate를 줄여서) Learning rate를 조절하는 것이다. 
Keras에서는 SGD(Stochastic Gradient Descent) optimizer를 default로 한다.

1. Constant Learning Rate
-  SGD optimizer
  - default: momentum = 0, decay rate = 0
  
    ```python
    keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
    ```
  
