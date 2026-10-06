# DNN Forward / Backward Propagation 구현 보고서

## 모델과 데이터
NumPy로 2차원 circles 데이터 1,000개를 직접 생성했다. 반지름 1.0과 0.4인 원에 표준편차 0.06의 가우시안 잡음을 더했다.
무작위로 학습 800개 / 검증 200개를 분할했으며 학습 데이터의 평균과 표준편차만 사용해 두 데이터를 표준화했다.
별도 최종 테스트 세트는 없으므로 기록한 성능은 검증 성능이다.

모델: 입력(2) → Dense(16) → ReLU → Dense(8) → Tanh → Dense(2) → Softmax Cross Entropy.
Dense는 총 3개, 은닉층은 2개, 학습 파라미터는 총 202개다.
모든 레이어는 (batch, features)를 유지한다. 가중치는 (입력차원, 출력차원), 편향은 (1, 출력차원)이다.
입력은 (N,2), 은닉층 출력은 (N,16), (N,8), 최종 logits은 (N,2)이다.

## 초기화·활성함수·손실함수
ReLU 앞 Dense는 He 초기화, Tanh와 출력 Dense는 Xavier 초기화를 적용했다. 편향은 0으로 초기화했다.
ReLU는 간단한 비선형성을 제공하며 양수 영역에서 기울기를 유지한다. Tanh는 매끄러운 비선형성을 제공한다.
두 클래스의 logits에 Softmax Cross Entropy를 적용하여 확률 정규화와 분류 손실을 함께 계산했다.
행별 최대 logits를 빼고 log-sum-exp를 사용하여 큰 값에서의 overflow를 방지했다.

## Forward와 직접 미분
Dense는 Z = XW+b를 계산하고 입력 X를 cache에 저장한다.
ReLU는 X>0 마스크를, Tanh는 활성화 출력을, 손실은 확률과 정답 label을 저장한다.

- Dense: dW = XᵀdZ, db = sum(dZ, axis=0, keepdims=True), dX = dZWᵀ.
- ReLU: dX = dout × 1[X>0]. X=0에서 미분값은 0으로 정의한다.
- Tanh: dX = dout × (1−tanh(X)²).
- Softmax CE: dlogits = (softmax(logits)−one_hot(y))/N.

배치 평균은 손실의 backward에서 한 번만 적용한다. Dense에서 다시 배치 크기로 나누지 않는다.
모델의 backward는 레이어를 역순으로 방문하며, SGD는 W←W−lr×dW, b←b−lr×db로 갱신한다.
자동미분 또는 딥러닝 프레임워크를 사용하지 않았다.

## Gradient checking
모델 seed=7, 데이터 seed=42의 학습 샘플 5개에 대해 모든 Dense의 W와 b를 검사했다.
중앙차분 g_num = (L(θ+ε)−L(θ−ε))/(2ε), ε=1e-5를 사용했다.
상대오차 = ||g_analytic−g_num||₂/(||g_analytic||₂+||g_num||₂+1e-12).
통과 기준은 1e-6이며 수치미분 후 원래 파라미터와 cache를 복원한다.

| 파라미터 | 상대오차 |
|---|---|
| layer_0.W | 5.710e-11 |
| layer_0.b | 3.929e-11 |
| layer_2.W | 6.888e-11 |
| layer_2.b | 5.641e-11 |
| layer_4.W | 2.101e-11 |
| layer_4.b | 2.287e-11 |

ReLU는 0에서 미분이 불연속이므로 수치미분에 사용한 입력에서 꺾이는 지점을 피해야 한다.
위 검사에서는 모든 오차가 기준보다 작아 통과했다.

## 실험 결과
200 epoch, batch size=32, 데이터 seed=42, 모델 seed=7, SGD를 사용했다.
같은 데이터·초기화·배치 순서에서 학습률만 0.01과 0.05로 바꾸었다.
매 epoch가 끝난 뒤 같은 파라미터 상태로 전체 학습/검증 loss와 accuracy를 계산했다.

| 학습률 | Train loss | Train accuracy | Val loss | Val accuracy |
|---|---|---|---|---|
| 0.01 | 0.009770 | 100.0% | 0.012252 | 100.0% |
| 0.05 | 0.001253 | 100.0% | 0.001908 | 100.0% |

두 학습률 모두 검증 정확도 100%에 도달했으며 0.05가 더 낮은 최종 loss를 기록했다.
각 epoch 로그는 CSV로 저장하며 아래 코드에서 loss/accuracy 곡선을 출력한다.
한 종류의 쉬운 toy 데이터와 단일 seed로 실험했으므로 일반 데이터의 성능으로 해석하지 않는다.

## 파일 구성과 실행 방법

| 파일 | 역할 |
|---|---|
| DNN_Submission.ipynb | 모듈 import, shape 확인, 테스트, 학습, 결과 시각화 |
| src/opt_utils.py | 데이터·미니배치, Dense/활성함수/손실, 모델 컨테이너 |
| src/optimization.py | Mini-batch SGD 학습 및 gradient checking |
| src/testCases.py | 재현 가능한 테스트 입력과 기대값 |
| src/test_utils.py | 수치미분·상대오차·shape/값 검사 |
| src/public_tests.py | 위 모듈을 실제 import하여 기능 검증 |
| README.md | 구현 설명과 실험 결과 |
| results/ | epoch별 CSV, summary.json, 학습 곡선 |

Google Colab에서 DNN_Submission.ipynb를 열고 런타임 → 모두 실행을 선택한다.
노트북만 업로드해도 첫 셀이 GitHub의 고정된 버전에서 필요한 .py 파일과 README를 받아 src 폴더를 구성한다.
ZIP 전체를 내려받아 로컬에서 실행할 때는 src 폴더와 노트북을 같은 프로젝트에 두면 다운로드가 필요 없다.
외부 데이터나 GPU는 필요 없다. 기존 교수님 파일을 별도로 덮어쓸 필요도 없다.
로컬 Python 패키지는 NumPy, Matplotlib이며 설치 명령은 `pip install -r requirements.txt`다.
기능 검증은 프로젝트 폴더에서 `python src/public_tests.py`로 실행한다.

첨부 Optimization.ipynb의 mini-batch/SGD 실습 흐름과 opt_utils.py/testCases.py/test_utils.py/public_tests.py의 모듈 역할을 참고했다.
기존 데이터 방향 (features,batch)은 사용자 조건 (batch,features)에 맞게 변경했다.
시험 조건에 따라 data.mat 대신 NumPy로 circles 데이터를 직접 생성한다.
