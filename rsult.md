# 모델 양자화 전후 성능 차이 분석 보고서

## 1. 실험 개요

본 실험은 동일한 행동 인식 모델을 세 가지 실행 형태로 비교한 결과이다.

| 모델 종류 | PYTORCH | TensorRT FP32 | TensorRT INT8 |
|---|---:|---:|---:|
| 모델 용량 | 358.5 MB | 120.4 MB | 약 31.1 MB |
| 정확도 | 66.06% | 64.71% | 66.06% |
| 윈도우당 평균 추론 시간 | 335.58 ms | 125.02 ms | 48.35 ms |
| 윈도우 처리 FPS | 2.98 | 8 | 20.68 |
| 평균 CPU 사용률 | 9.88% | 5.25% | 10.99% |
| 평균 GPU 사용률 | 87.41% | 89.81% | 69.40% |
| 평균 메모리 사용량 | 10514/62841 MB | 9738/62841 MB | 9712/62841 MB |
| 평균 전력 사용량 | 4808 mW | 4901 mW | 4350 mW |

평가 단위는 단일 프레임이 아니라 `16프레임 입력 윈도우`이다. 모델 학습 코드와 추론 설정 모두 입력을 `3 x 16 x 112 x 112` 형태로 사용한다. 따라서 `윈도우당 평균 추론 시간`은 1프레임 처리 시간이 아니라 16프레임 클립 하나를 추론하는 평균 시간이다.

## 2. 평가 방식

본 프로젝트의 원본 validation XML에는 `normal` 라벨이 명시적으로 존재하지 않는다. 따라서 이벤트 구간 밖을 무조건 `normal`로 간주하면 정확도가 왜곡된다. 이를 피하기 위해 최종 평가는 XML에 라벨링된 이벤트 구간만 대상으로 수행하였다.

평가 기준은 다음과 같다.

| 항목 | 내용 |
|---|---|
| 입력 단위 | 16프레임 sliding window |
| 이벤트 정답 | XML의 `*_start`부터 `*_end`까지 |
| 평가 포함 | 예측 기준 프레임이 이벤트 구간 안에 있는 윈도우 |
| 평가 제외 | XML 이벤트 구간 밖의 unlabeled 윈도우 |
| 정확도 | 이벤트 구간 내 정답 윈도우 수 / 이벤트 구간 내 전체 평가 윈도우 수 |

Python 앱은 `base_model/inference.py`에서 validation 영상을 한 번 읽어 캐싱한 후 sliding window를 만들어 평가한다. C++ 앱은 `app/run.sh`에서 DeepStream 앱을 실행하고, 출력된 예측 프레임이 XML 이벤트 구간에 포함되는지 확인하여 동일한 방식으로 정확도를 계산한다.

## 3. 소스 코드 기준 모델 구조와 전처리

학습 및 Python 추론은 `base_model/dataset.py`와 `base_model/inference.py` 기준으로 동작한다.

핵심 전처리는 다음과 같다.

- OpenCV로 영상을 읽음
- BGR을 RGB로 변환
- `112 x 112`로 resize
- `frames.astype(np.float32) / 255.0`
- 평균 `0.45`, 표준편차 `0.225`로 normalization
- 텐서 형태는 `C x T x H x W`

DeepStream C++ 앱은 `app/configs/config_preprocess_action.txt`에서 같은 전처리를 맞춘다.

```text
network-input-shape=1;3;16;112;112
processing-width=112
processing-height=112
channel-scale-factors=0.0174291939;0.0174291939;0.0174291939
channel-mean-offsets=114.75;114.75;114.75
stride=1
```

위 값은 PyTorch 전처리 `(pixel / 255 - 0.45) / 0.225`와 동등하도록 설정된 것이다. 따라서 FP32/INT8 TensorRT 엔진은 Python 모델과 같은 입력 의미를 갖도록 구성되어 있다.

## 4. 모델 용량 차이 원인

모델 용량은 다음 순서로 감소하였다.

```text
PYTORCH 358.5 MB
FP32 engine 120.4 MB
INT8 engine 약 31.1 MB
```

### 4.1 PyTorch `.pth`가 가장 큰 이유

PyTorch 체크포인트는 단순한 추론 엔진이 아니라 학습 과정에서 저장된 state dict 구조를 포함한다. 일반적으로 다음과 같은 부가 정보가 포함될 수 있다.

- 모델 파라미터
- optimizer 또는 학습 관련 상태
- epoch, history 등 메타데이터
- PyTorch 직렬화 오버헤드

반면 TensorRT engine은 배포용 추론 엔진이다. 학습 재개에 필요한 정보가 아니라, 추론에 필요한 그래프와 weight 중심으로 최적화되어 저장된다. 이 때문에 같은 FP32라도 PyTorch `.pth`보다 FP32 engine이 훨씬 작다.

### 4.2 INT8 engine이 가장 작은 이유

FP32는 weight 하나를 32비트로 저장하고, INT8은 8비트로 저장한다. 단순 계산으로도 weight 저장 단위가 1/4로 줄어든다.

실험 결과도 이 경향과 일치한다.

```text
FP32 engine 120.4 MB
INT8 engine 약 31.1 MB
```

즉 INT8 engine은 FP32 engine 대비 약 25.8% 수준이다. 이는 32비트에서 8비트로 줄어드는 양자화 효과와 잘 맞는다.

## 5. 정확도 차이 분석

정확도는 다음과 같다.

```text
PYTORCH: 66.06%
FP32:    64.71%
INT8:    66.06%
```

일반적으로 양자화는 정밀도 손실 때문에 정확도가 떨어질 수 있다. 그러나 본 결과에서는 INT8이 PyTorch와 동일한 66.06%를 기록했고, FP32 TensorRT가 오히려 약간 낮은 64.71%를 기록했다.

가능한 원인은 다음과 같다.

### 5.1 평가 데이터 수가 작아 1개 윈도우 차이가 정확도에 크게 반영됨

표의 정확도는 이벤트 구간 window 기준으로 계산된다. 전체 평가 윈도우 수가 많지 않으면 몇 개의 예측 차이만으로도 1% 내외의 정확도 차이가 발생할 수 있다.

따라서 FP32의 64.71%와 INT8/PyTorch의 66.06% 차이는 모델 구조의 본질적 성능 차이라기보다, 일부 boundary window나 confidence가 낮은 window에서 argmax가 달라진 결과일 가능성이 크다.

### 5.2 TensorRT 변환 과정에서 연산 구현이 달라짐

PyTorch는 `torchvision.models.video.r2plus1d_18` 기반으로 추론하고, TensorRT는 ONNX/engine 변환 후 실행한다. 같은 FP32라도 다음 요소가 달라질 수 있다.

- convolution 알고리즘 선택
- layer fusion
- floating point 연산 순서
- resize/preprocess 구현 차이
- TensorRT 최적화 tactic 선택

부동소수점 연산은 결합법칙이 완전히 보장되지 않기 때문에, 작은 수치 차이가 softmax argmax를 바꿀 수 있다. 특히 행동 클래스 간 score가 비슷한 window에서는 미세한 차이로 예측 클래스가 바뀔 수 있다.

### 5.3 INT8 calibration이 현재 데이터 분포와 잘 맞았을 가능성

INT8은 `quantized/best_model_int8.engine`과 calibration cache를 기반으로 실행된다. calibration 데이터가 실제 validation 영상 분포와 유사했다면 INT8 scale이 모델 활성값 범위를 잘 표현했을 가능성이 있다.

그 결과 INT8로 줄였음에도 정확도 손실이 거의 없거나, 일부 애매한 window에서는 오히려 PyTorch와 같은 argmax를 선택했을 수 있다.

### 5.4 normal 라벨 문제를 회피한 event-only 평가의 영향

현재 평가는 XML 이벤트 구간만 포함한다. 즉 `normal` 클래스 성능은 직접 평가하지 않는다. 이 방식은 현재 데이터셋 구조에서는 정당하지만, 전체 5클래스 분류 성능을 완전히 대표하지는 않는다.

따라서 본 정확도는 다음 의미로 해석해야 한다.

```text
라벨링된 이상행동 이벤트 구간에서 해당 이벤트 클래스를 맞춘 비율
```

`normal`까지 포함한 일반적인 5클래스 정확도와는 다르다.

## 6. 추론 속도 차이 분석

추론 시간은 다음과 같다.

```text
PYTORCH: 335.58 ms/window
FP32:    125.02 ms/window
INT8:     48.35 ms/window
```

처리량은 다음과 같다.

```text
PYTORCH: 2.98 window/s
FP32:    8 window/s
INT8:    20.68 window/s
```

### 6.1 PyTorch가 가장 느린 이유

Python 앱은 PyTorch eager execution 기반이다. 모델 forward 자체는 GPU를 사용하지만, 다음 오버헤드가 존재한다.

- Python 루프
- PyTorch eager dispatch overhead
- CPU에서 tensor slice 생성
- CPU to GPU tensor transfer
- CUDA kernel launch overhead

`base_model/inference.py`는 영상을 한 번 읽어 캐싱하도록 개선되어 반복 디코딩 비용은 줄었지만, 여전히 window마다 Python에서 모델을 호출한다. 작은 batch size 1의 window 추론에서는 이러한 호출 오버헤드가 상대적으로 크게 작용한다.

### 6.2 TensorRT FP32가 PyTorch보다 빠른 이유

TensorRT FP32 engine은 추론 그래프가 최적화되어 있다. 주요 차이는 다음과 같다.

- layer fusion
- 불필요한 framework overhead 제거
- Jetson GPU에 맞는 kernel/tactic 선택
- DeepStream pipeline 내에서 GPU 메모리 기반 처리

이 때문에 FP32 정밀도를 유지해도 PyTorch보다 약 2.68배 빠르다.

```text
335.58 / 125.02 ≈ 2.68
```

### 6.3 INT8이 가장 빠른 이유

INT8은 연산량과 메모리 대역폭 요구량이 줄어든다. Jetson AGX 계열 GPU/DLA/Tensor Core 최적화 경로에서는 INT8 연산이 FP32보다 훨씬 높은 처리량을 낼 수 있다.

실험 결과 INT8은 PyTorch 대비 약 6.94배, TensorRT FP32 대비 약 2.59배 빠르다.

```text
335.58 / 48.35 ≈ 6.94
125.02 / 48.35 ≈ 2.59
```

따라서 본 모델은 INT8 양자화로 속도 이득을 매우 크게 얻었다.

## 7. CPU/GPU/메모리/전력 차이 분석

### 7.1 CPU 사용률

```text
PYTORCH: 9.88%
FP32:    5.25%
INT8:   10.99%
```

FP32 TensorRT의 CPU 사용률이 가장 낮다. DeepStream/TensorRT 파이프라인이 GPU 중심으로 동작하고, Python interpreter overhead가 없기 때문이다.

INT8의 CPU 사용률이 FP32보다 높게 보이는 이유는 추론 자체가 빨라지면서 영상 디코딩, 파이프라인 제어, 로그 처리, 프로세스 반복 실행 등 CPU 주변 작업의 상대적 비중이 증가했기 때문으로 볼 수 있다. 즉 INT8 연산이 가벼워진 만큼 CPU가 다음 입력을 준비하는 역할이 더 눈에 띄게 측정된다.

### 7.2 GPU 사용률

```text
PYTORCH: 87.41%
FP32:   89.81%
INT8:   69.40%
```

FP32는 무거운 FP32 연산을 수행하기 때문에 GPU 사용률이 높다. PyTorch도 GPU를 많이 사용하지만 framework overhead와 CPU-GPU 전송이 있어 FP32 TensorRT와 양상이 다르다.

INT8은 더 빠르고 가벼운 연산을 사용하므로 GPU가 같은 작업을 더 짧은 시간에 끝낸다. 그 결과 평균 GPU 사용률은 낮아졌지만 처리량은 오히려 가장 높다.

즉 INT8의 낮은 GPU 사용률은 성능 저하가 아니라, 연산 효율이 좋아져 GPU 점유 시간이 줄어든 결과로 해석할 수 있다.

### 7.3 메모리 사용량

```text
PYTORCH: 10514 MB
FP32:    9738 MB
INT8:    9712 MB
```

PyTorch가 가장 많은 메모리를 사용한다. 이유는 다음과 같다.

- PyTorch runtime과 CUDA context
- model object와 tensor 관리 오버헤드
- Python 객체 및 numpy/tensor cache
- framework 내부 allocator

TensorRT engine은 추론 전용 런타임이므로 PyTorch보다 메모리 사용량이 낮다. FP32와 INT8의 평균 메모리 차이가 크지 않은 이유는 전체 시스템 메모리에는 모델 weight 외에도 DeepStream, GStreamer, CUDA context, decoder, buffer pool 등이 포함되기 때문이다. 모델 파일 크기는 크게 줄어도 런타임 전체 메모리에서 차지하는 비중은 제한적일 수 있다.

### 7.4 전력 사용량

```text
PYTORCH: 4808 mW
FP32:    4901 mW
INT8:    4350 mW
```

INT8의 평균 전력이 가장 낮다. 이는 INT8 추론이 FP32보다 적은 연산량과 낮은 메모리 대역폭을 요구하기 때문이다.

FP32 TensorRT는 GPU 사용률이 가장 높고 FP32 연산을 수행하므로 전력 사용량도 가장 높게 측정되었다. PyTorch는 FP32 TensorRT보다 느리지만 runtime overhead와 GPU 사용이 함께 발생하여 전력 사용량이 큰 편이다.

## 8. 왜 INT8이 정확도는 유지하면서 속도와 전력은 개선되었는가

본 실험의 핵심 결과는 다음이다.

```text
INT8은 PyTorch와 같은 66.06% 정확도를 유지하면서,
윈도우당 평균 추론 시간을 335.58ms에서 48.35ms로 줄였다.
```

가능한 원인은 다음과 같다.

1. 모델이 INT8 양자화에 비교적 강건하다.
2. calibration 데이터가 실제 validation 분포와 잘 맞았다.
3. 평가가 event-only이므로 normal 클래스 혼동의 영향이 제거되었다.
4. TensorRT가 layer fusion과 kernel 최적화를 수행했다.
5. INT8 weight/activation 사용으로 메모리 대역폭과 연산량이 감소했다.

따라서 현재 실험 조건에서는 INT8 엔진이 가장 적합한 배포 후보이다.

## 9. 주의할 점과 한계

### 9.1 본 정확도는 normal 포함 전체 정확도가 아니다

현재 XML에는 normal 구간이 명시적으로 라벨링되어 있지 않다. 따라서 본 보고서의 정확도는 event-only accuracy이다.

즉 다음 성능을 의미한다.

```text
라벨링된 이상행동 구간에서 해당 이상행동 클래스를 맞추는 능력
```

normal을 포함한 실제 운영 환경의 전체 정확도를 보려면 normal 라벨이 명시된 데이터셋 또는 정상 클립 기준의 별도 평가셋이 필요하다.

### 9.2 PyTorch와 DeepStream의 측정 경로가 완전히 같지는 않다

두 앱의 평가 논리는 동일하게 맞추었지만 실행 경로는 다르다.

- Python: PyTorch 모델 forward 중심
- C++: DeepStream + TensorRT + GStreamer pipeline

따라서 총 실행 시간은 애플리케이션 구조의 영향을 받는다. 모델 자체의 연산 성능 비교에는 `윈도우당 평균 추론 시간`과 `윈도우 처리 FPS`가 더 유용하다.

### 9.3 리소스 사용량은 tegrastats 평균값이다

CPU/GPU/메모리/전력은 `tegrastats` 로그를 평균 낸 값이다. 샘플 수가 적거나 실행 시간이 짧으면 순간 부하의 영향을 받을 수 있다.

## 10. 결론

본 실험에서 INT8 양자화는 모델 용량, 추론 속도, 전력 효율 측면에서 가장 좋은 결과를 보였다.

핵심 비교는 다음과 같다.

| 비교 | 결과 |
|---|---|
| 모델 용량 | PyTorch 358.5 MB → INT8 약 31.1 MB |
| 정확도 | PyTorch 66.06% → INT8 66.06% |
| 평균 추론 시간 | PyTorch 335.58ms → INT8 48.35ms |
| 처리량 | PyTorch 2.98 FPS → INT8 20.68 FPS |
| 전력 | PyTorch 4808mW → INT8 4350mW |

따라서 현재 코드와 평가셋 기준에서는 INT8 TensorRT 엔진이 정확도 손실 없이 가장 높은 추론 성능과 가장 낮은 전력 사용량을 제공한다. Jetson AGX 환경에 배포하기 위한 최종 후보로는 INT8 엔진이 가장 타당하다.

