# NumPy DNN의 Forward / Backward Propagation 직접 구현

| 항목 | 내용 |
|---|---|
| 제출자 | 이름을 입력하세요 |
| 학번 | 학번을 입력하세요 |
| 과목 | 딥러닝 + 머신러닝 |
| 평가 주제 | Dense 기반 DNN의 순전파·역전파 직접 구현 |

## 1. 제출 파일과 실행 방법

첨부된 `Optimization.ipynb`의 초기화 → 미니배치 → forward → loss → backward → 업데이트 순서를 활용하였다. 평가 조건에 맞추어 NumPy toy 데이터, `(batch, features)` 배열, 레이어 객체, 직접 역전파, 수치미분 검사를 구현하였다.

제출용 ZIP을 풀면 다음 **3개 파일**이 포함된다.

```text
DNN_Submission/
├── Optimization.ipynb     # 소스 모듈을 불러와 실행하는 실습 노트북
├── README.md             # 구현 설명과 실험 결과
└── src/
    └── opt_utils.py       # 노트북이 실제로 import하는 NumPy 구현 소스
```

1. GitHub의 [DNN_Submission.zip](https://github.com/piaa1004/python/raw/refs/heads/main/DNN_Submission.zip)을 다운로드하고 압축을 푼다.
2. Google Colab에서 **파일 → 노트북 업로드**로 `Optimization.ipynb`를 연다.
3. Python 3 / CPU 런타임에서 **런타임 → 모두 실행**을 선택한다.
4. 데이터·shape 확인, gradient checking, 학습 로그, 곡선, 최종 성능을 확인한다.

`src/opt_utils.py`는 노트북 실행에 필요한 실제 구현 모듈이다. 노트북은 이 파일에서 함수와 클래스를 import하여 사용하며, 같은 구현을 노트북 안에 중복 정의하지 않는다. ZIP을 푼 폴더에서 실행하면 동봉된 파일을 사용한다. Colab에 노트북만 업로드하면 첫 셀이 GitHub의 고정된 버전에서 `src/opt_utils.py`를 자동으로 다운로드한 뒤 import한다. 따라서 Colab에서는 사용자가 따로 파일을 업로드하거나 import 코드를 추가할 필요가 없다. 첫 셀의 `Imported module: src/opt_utils.py` 출력으로 연결을 확인할 수 있다.

소스 구현을 수정하려면 `src/opt_utils.py`를 편집한다. 이미 모듈을 import한 뒤 파일을 수정했다면 런타임을 다시 시작하고 전체 셀을 실행한다. 자동 다운로드는 파일이 없을 때만 수행하므로 기존 소스를 덮어쓰지 않는다. 로컬 Jupyter에서는 `Optimization.ipynb`와 `src/`가 함께 있는 폴더를 작업 폴더로 사용하고 NumPy와 Matplotlib을 설치한다.

NumPy와 Matplotlib을 사용하며 GPU는 필요하지 않다. 제출된 노트북은 전체 셀을 실행하여 결과를 확인하였다. 제출 전 위 표에 이름과 학번을 입력하고 `DNN_Submission` 폴더를 제출한다.

## 2. 원본 실습을 활용한 부분

| 원본 함수 | 이번 구현 |
|---|---|
| `update_parameters_with_gd()` | `W -= learning_rate * dW`, `b -= learning_rate * db` 및 파라미터·gradient 딕셔너리 구조를 유지한다. 모델의 `step()`에서 실제로 호출한다. |
| `random_mini_batches()` | 함수 이름을 유지하고 `get_batches()`를 호출한다. 입력과 정답을 같은 순서로 섞는다. |
| `initialize_parameters(layers_dims)` | 차원 설정 방식을 유지한다. 레이어 리스트로 구성한 모델 객체를 반환한다. |
| `forward_propagation()`, `backward_propagation()` | 함수 이름을 유지하며 모델의 `forward()`와 `backward()`를 호출한다. |
| `compute_cost()`, `predict()`, `load_dataset()` | 실습의 함수 이름을 유지하고 새 손실함수·데이터·배치 방향에 맞게 구현한다. |
| `model()` | epoch 안에서 미니배치별 순전파 → 손실 → 역전파 → 업데이트 순서를 유지한다. |

원본의 `(features, batch)`는 평가 조건에 맞추어 **`(batch, features)`**로 변경하였다. 정답은 클래스 번호를 담은 `(batch,)` 배열이다. 평가에서 NumPy로 toy 데이터를 직접 생성하도록 요구하므로 원본 데이터 로딩을 circles 생성으로 바꾸었다. 따라서 원본 `data.mat`와 테스트 보조 파일은 이번 노트북 실행에 필요하지 않다.

## 3. 데이터 파이프라인

`load_data(n_samples=1200, noise=0.12, seed=3)`가 두 동심원 데이터를 직접 생성한다. 클래스별 600개이며 반지름은 1.0과 0.45이다. 각도는 `[0, 2π)`에서 균등하게 뽑고 좌표에 표준편차 0.12의 가우시안 잡음을 더한다.

`split_data()`로 클래스 비율을 유지하며 먼저 train pool:test를 **8:2**로 나눈 뒤, train pool의 20%를 validation으로 분리한다.

| 데이터 | 샘플 수 | 전체 비율 | 용도 |
|---|---:|---:|---|
| Train | 768 | 64% | 학습·gradient 계산 |
| Validation | 192 | 16% | 학습률 선택 |
| Test | 240 | 20% | 선택 후 최종 평가 |

`standardize()`는 train의 평균·표준편차만 구하고 동일한 통계로 세 데이터를 표준화한다. 계산식은 `(X - train_mean) / train_std`이다. Validation과 test의 통계는 사용하지 않는다.

`get_batches(X, y, batch_size, shuffle=True)`는 입력과 정답에 동일한 인덱스를 적용하고 마지막 작은 배치도 반환한다. 이번 학습은 배치 크기 32로, epoch마다 24개 배치를 사용한다.

## 4. 모델 구조·초기화·선택 이유

```text
입력 (B, 2)
 → Dense(2, 16) → ReLU
 → Dense(16, 8) → Tanh
 → Dense(8, 2) → Softmax Cross Entropy
```

`B`는 배치 크기이다. **Dense 3개, 은닉층 2개**이며 출력 Dense는 logits을 반환한다. Softmax는 손실 모듈에서 계산한다.

| 레이어 | W shape | b shape | 출력 shape | 가중치 초기화 | 파라미터 수 |
|---|---|---|---|---|---:|
| Dense 1 | `(2, 16)` | `(1, 16)` | `(B, 16)` | He | 48 |
| Dense 2 | `(16, 8)` | `(1, 8)` | `(B, 8)` | Xavier | 136 |
| Dense 3 | `(8, 2)` | `(1, 2)` | `(B, 2)` | Xavier | 18 |
| 합계 | | | | | **202** |

He는 `N(0, 1) * sqrt(2 / fan_in)`, Xavier는 `N(0, 1) * sqrt(1 / fan_in)`으로 가중치를 초기화한다. 첫 번째 Dense에는 ReLU에 맞춘 He를, 나머지 Dense에는 Xavier를 적용한다. bias는 모두 0으로 초기화한다.

ReLU는 양수 입력의 gradient를 전달하며 계산이 간단하다. Tanh는 `[-1, 1]` 출력과 `1 - tanh(x)^2` 미분을 사용해 두 번째 비선형 은닉층을 구성한다. 두 클래스의 logits과 정수 정답을 사용하므로 Softmax Cross Entropy를 선택하였다.

## 5. 순전파·역전파와 캐시

`DenseLayer`는 `W`, `b`, `dW`, `db`를 보관하고 forward의 입력과 출력 shape를 캐싱한다.

```text
Dense forward:  out = X @ W + b
Dense backward: dW = X.T @ dout
                db = sum(dout, axis=0, keepdims=True)
                dX = dout @ W.T
ReLU forward:   out = max(0, X)
ReLU backward:  dX = dout * (X > 0)
Tanh forward:   out = tanh(X)
Tanh backward:  dX = dout * (1 - out**2)
```

ReLU는 순전파 입력, Tanh는 순전파 출력을 캐싱한다. ReLU의 0에서의 미분은 0으로 정의한다. 모든 모듈에서 순전파와 gradient의 shape를 검사하며 `dW`, `db`, `dX`는 각각 `W`, `b`, `X`의 shape와 일치한다.

```text
shifted   = logits - max(logits, axis=1, keepdims=True)
log_probs = shifted - log(sum(exp(shifted), axis=1, keepdims=True))
probs     = exp(log_probs)
loss      = -mean(log_probs[샘플 인덱스, 정답 클래스])
dlogits   = (probs - one_hot(y)) / B
```

`SoftmaxCrossEntropy`는 확률과 정답을 캐싱한다. Log-sum-exp 계산으로 큰 logits의 수치 불안정을 줄인다. 실제 backward에서는 정답 클래스 위치에서 1을 빼며, **배치 평균은 손실의 backward에서 한 번만** 적용한다. Dense에서 다시 배치 크기로 나누지 않는다.

`DeepNeuralNetwork`는 레이어 리스트를 순서대로 forward하고 역순으로 backward한다. `step(learning_rate)`에서 `W -= learning_rate * dW`, `b -= learning_rate * db`를 적용한다. 모든 미분은 NumPy로 직접 구현하며 자동미분을 사용하지 않는다.

## 6. Gradient checking 결과

학습 전 train 샘플 5개를 사용해 모든 Dense의 **파라미터 202개 전체**와 입력의 **10개 값**을 검사하였다. 중앙차분과 직접 구현한 gradient를 비교한다.

```text
numeric = [J(theta + epsilon) - J(theta - epsilon)] / (2 * epsilon)
relative_error = ||analytic - numeric||₂ / (||analytic||₂ + ||numeric||₂ + 1e-12)
epsilon = 1e-5, 통과 기준: relative_error < 1e-6
```

검사 후 변경한 값과 캐시·gradient를 복원한다. 검사 배치의 ReLU 입력 최소 절댓값은 `0.0010169582`로 0 근방을 피하는 확인 기준 `100 * epsilon = 0.001`보다 크다.

| 검사 항목 | Shape | 검사 수 | 상대 오차 | 최대 절대 오차 |
|---|---|---:|---:|---:|
| W1 | `(2, 16)` | 32 | `8.168498e-11` | `1.194633e-11` |
| b1 | `(1, 16)` | 16 | `6.255803e-11` | `1.587930e-11` |
| W2 | `(16, 8)` | 128 | `1.290690e-10` | `2.016695e-10` |
| b2 | `(1, 8)` | 8 | `3.504799e-11` | `1.118364e-11` |
| W3 | `(8, 2)` | 16 | `1.579987e-11` | `7.398082e-12` |
| b3 | `(1, 2)` | 2 | `5.668047e-11` | `1.148573e-11` |
| X → dX | `(5, 2)` | 10 | `1.148849e-10` | `1.267906e-11` |

7개 검사 항목이 모두 통과하였다. 최대 상대 오차는 약 **`1.29e-10`**이며, 이 검사 지점에서 수치미분과 직접 구현한 역전파가 일치함을 확인하였다.

## 7. 학습 설정과 실험 결과

**Mini-batch SGD / 배치 32 / 300 epochs**로 학습률 **0.01과 0.1**을 비교하였다. 데이터·모델 seed는 3, 분할 seed는 10과 11, 배치 셔플 seed는 10이다. 두 실험은 같은 데이터·초기 가중치·배치 순서를 사용하고 학습률만 바꾸었다.

각 epoch 종료 시 전체 train·validation의 loss와 accuracy를 평가한다. 전체 300 epoch 지표는 메모리의 `history`에 기록하며 노트북에서 학습 곡선을 출력한다. 콘솔에는 첫 epoch와 50 epoch 간격의 로그를 출력한다. `batch_loss`는 업데이트 중 배치 손실의 가중평균이며 아래 train loss는 epoch 종료 후 다시 평가한 값이다.

학습 전 공통 train loss는 **0.712891**이었다. 전체 실행에서 얻은 최종 결과는 다음과 같다.

| 학습률 | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 0.01 | 0.027470 | 99.6094% | 0.047796 | 97.9167% |
| 0.1 | 0.007302 | 99.8698% | 0.047773 | 97.9167% |

50 epoch의 train loss는 학습률 0.01에서 **0.158141**, 0.1에서 **0.019931**이었다. 이번 실행에서는 0.1이 초기에 더 빠르게 손실을 줄였다. 300 epoch의 validation 정확도는 같고 validation loss 차이는 매우 작으므로 일반화 성능이 크게 향상되었다고 해석하지 않는다.

300 epoch 종료 시 **validation loss가 더 작은 학습률 0.1**을 선택하였다. 선택 후 test를 한 번 평가한 결과 **test loss 0.095187, test accuracy 97.5%**를 얻었다. Test 성능은 학습률 선택에 사용하지 않았다. 노트북 출력에서 학습 곡선과 선택한 모델의 분류 경계도 확인할 수 있다.

## 8. 평가 조건 확인

- Dense 3개와 은닉층 2개, He/Xavier 가중치 초기화를 적용하였다.
- `(batch, features)` shape와 각 모듈의 forward 캐시를 유지하였다.
- Dense의 `dW`, `db`, `dX`, ReLU·Tanh·Softmax Cross Entropy 미분을 직접 구현하였다.
- NumPy circles 생성, train/test 분할, validation 분리, train 통계 표준화를 구현하였다.
- 미니배치 iterator, SGD, epoch별 train·validation 지표, 최종 test 평가를 구현하였다.
- 수치미분 검사 오차와 학습률 비교 결과를 노트북 및 README에 제시하였다.
