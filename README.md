# Deep Learning G12 Repository Guide

이 저장소는 행동 인식 모델의 학습, TensorRT 양자화, DeepStream 기반 실행 애플리케이션을 나누어 관리합니다. 주요 디렉토리는 `base_model`, `quantized_model`, `app` 세 부분입니다.

## 전체 구조

```text
.
├── app/              # DeepStream/GStreamer 기반 행동 인식 실행 애플리케이션
├── base_model/       # PyTorch 기반 원본 모델 학습 및 추론 코드
└── quantized_model/  # 학습된 모델을 TensorRT 엔진으로 변환/양자화하는 코드
```

## `base_model/`

PyTorch로 행동 인식 모델을 학습하고, 학습된 체크포인트로 단일 영상 추론을 수행하는 원본 모델 코드 디렉토리입니다. 현재는 별도 하위 디렉토리 없이 학습에 필요한 Python 파일들이 평평하게 배치되어 있습니다.

주요 파일:

- `config.py`
  - 데이터셋 경로, 체크포인트 저장 경로, 프레임 수, 클래스 수, batch size, epoch, learning rate 등 학습/추론 공통 설정을 정의합니다.
- `dataset.py`
  - `ActionDataset`을 정의합니다.
  - `train`, `val`, `test` split 아래의 클래스별 `.mp4` 클립을 읽고, 모델 입력 형태인 `(C, T, H, W)` 텐서로 전처리합니다.
  - 클래스는 `abandon`, `fight`, `broken`, `theft`, `normal`로 매핑됩니다.
- `train.py`
  - `torchvision.models.video.r2plus1d_18` 기반 모델을 학습합니다.
  - 클래스 불균형을 줄이기 위해 `WeightedRandomSampler`를 사용합니다.
  - validation accuracy 기준으로 best checkpoint와 학습 history를 저장합니다.
- `inference.py`
  - 저장된 PyTorch checkpoint를 로드해 단일 영상에 대한 행동 클래스를 예측합니다.
  - 학습 때와 동일한 프레임 샘플링 및 정규화 과정을 사용합니다.

## `quantized_model/`

`base_model`에서 학습한 PyTorch checkpoint를 ONNX 및 TensorRT engine으로 변환하는 디렉토리입니다. 배포나 DeepStream 실행에 사용할 FP16/INT8 엔진 생성 작업(양자화를 의미합니다)을 담당합니다.

하위 디렉토리:

- `fp16/`
  - FP16 TensorRT 엔진 생성을 위한 스크립트와 설정 파일을 보관합니다.
  - `quantize_trtexec_fp16.py`
    - PyTorch checkpoint를 ONNX로 export한 뒤 `trtexec`으로 TensorRT engine을 생성합니다.
    - 기본 precision은 FP16이며, 옵션에 따라 INT8 실행에도 일부 재사용할 수 있습니다.
  - `trtexec_config_fp16.json`
    - checkpoint, ONNX, engine 출력 경로, batch size, workspace 등 `trtexec` 실행 설정을 담는 JSON 파일입니다.

- `int8/`
  - INT8 TensorRT 엔진 생성을 위한 캘리브레이션 관련 코드와 실행 스크립트를 보관합니다.
  - `README.md`
    - INT8 캘리브레이션 절차와 산출물 경로를 설명합니다.
  - `run_cpp.sh`
    - C++ 기반 INT8 calibrator를 실행하기 위한 스크립트입니다.
  - `scripts/`
    - INT8 캘리브레이션 입력 데이터를 준비하는 Python 스크립트를 보관합니다.
    - `prepare_calib_tensors.py`는 `.mp4` 클립을 모델 입력 형식에 맞는 binary tensor 파일로 변환합니다.
  - `tensorrt_int8_calibrator/`
    - TensorRT INT8 calibration 및 engine build를 수행하는 C++ 패키지입니다.
    - CMake와 clangd 설정을 포함하며, `build-debug/compile_commands.json`을 통해 clangd가 include path와 compile option을 인식합니다.
    - 주요 모듈:
      - `binary_tensor_calibrator.*`: binary tensor 파일을 읽어 TensorRT entropy calibrator에 공급합니다.
      - `int8_engine_builder.*`: ONNX 모델을 파싱하고 INT8 TensorRT engine을 빌드합니다.
      - `calibrator_options.*`: CLI 옵션과 기본 경로를 관리합니다.
      - `trt_logger.*`: TensorRT 로그 출력을 처리합니다.
      - `file_utils.*`, `trt_utils.*`: 파일 입출력과 CUDA/TensorRT 유틸리티를 제공합니다.
    - `build/`, `build-debug/`, `build-release/`는 CMake 빌드 산출물 디렉토리입니다.

## `app/`

양자화된 TensorRT engine을 DeepStream/GStreamer pipeline에서 실행하기 위한 C++ 애플리케이션 디렉토리입니다. 실시간 또는 파일 기반 영상 입력을 받아 행동 인식 결과를 처리하는 실행 단계의 코드를 포함합니다.

하위 디렉토리:

- `configs/`
  - DeepStream pipeline 실행에 필요한 설정 파일을 보관합니다.
  - `config_preprocess_action.txt`
    - 행동 인식 모델 입력 전처리 설정을 정의합니다.
  - `config_infer_primary_action.txt`
    - primary inference 설정을 정의합니다. TensorRT engine, label, batch 등 추론 관련 옵션이 이 파일에 들어갑니다.
  - `labels.txt`
    - 모델 출력 index와 행동 클래스 이름을 매핑합니다.

- `src/`
  - DeepStream 실행 애플리케이션의 C++ 소스 코드입니다.
  - `main.cpp`
    - GStreamer를 초기화하고 앱 설정을 읽은 뒤 action recognition pipeline을 실행합니다.
  - `config/`
    - 실행 인자와 앱 설정을 읽고 관리하는 `AppConfig` 모듈입니다.
  - `video_input/`
    - 영상 입력 source bin 구성을 담당합니다.
  - `action_pipeline/`
    - DeepStream/GStreamer 기반 행동 인식 pipeline 생성과 실행을 담당합니다.
  - `prediction/`
    - 추론 결과 metadata를 읽고 후처리/출력하는 모듈입니다.

- `build/`
  - CMake 빌드 결과물이 생성되는 디렉토리입니다.
  - `deepstream_action_predict` 실행 파일과 `compile_commands.json`이 생성됩니다.

주요 파일:

- `CMakeLists.txt`
  - DeepStream, GStreamer, GLib include/library를 연결해 `deepstream_action_predict` 실행 파일을 빌드합니다.
- `.clangd`
  - clangd가 CMake의 `compile_commands.json`을 사용하도록 설정합니다.
- `build.sh`
  - 앱을 빌드하는 스크립트입니다.
- `run.sh`
  - 빌드된 DeepStream 행동 인식 애플리케이션을 실행하는 스크립트입니다.
- `clean.sh`
  - 앱 빌드 산출물을 정리하는 스크립트입니다.

## 작업 흐름

1. `base_model/`에서 PyTorch 모델을 학습하고 checkpoint를 생성합니다.
2. `quantized_model/fp16/` 또는 `quantized_model/int8/`에서 checkpoint를 ONNX/TensorRT engine으로 변환합니다.
3. `app/configs/`에서 생성된 engine과 label 경로를 설정합니다.
4. `app/`의 DeepStream C++ 애플리케이션을 빌드하고 실행합니다.
