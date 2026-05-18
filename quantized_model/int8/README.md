# TensorRT INT8 Calibration

이 디렉토리는 INT8 버전의 TensorRT 엔진(`.engine`)과 캘리브레이션 캐시 파일(`.cache`)을 빌드하기 위한 공간입니다.

## Run

```bash
cd /root/deeplearning_g12/quantized_model/calib_cash
./run_cpp.sh
```

## Structure

- `scripts/prepare_calib_tensors.py`
  - 캘리브레이션용 `.mp4` 영상 클립들을 모델 입력 포맷에 맞는 텐서 파일(`.bin`)로 변환하는 스크립트
- `tensorrt_int8_calibrator/`
  - C++ 기반의 TensorRT 캘리브레이션 실행 애플리케이션 디렉토리
- `tensorrt_int8_calibrator/binary_tensor_calibrator.*`
  - 준비된 바이너리 텐서 파일들을 모델에 피딩하여 엔트로피 캘리브레이션(Entropy Calibration)을 수행하는 모듈
- `tensorrt_int8_calibrator/int8_engine_builder.*`
  - ONNX 모델을 파싱하고 최종 INT8 엔진을 빌드하는 모듈
- `tensorrt_int8_calibrator/calibrator_options.*`
  - CLI(커맨드 라인 인터페이스) 옵션 파싱 및 기본 파일 경로들을 관리하는 모듈
- `tensorrt_int8_calibrator/trt_logger.*`
  - TensorRT 내부 로그 출력 및 관리를 위한 로거(Logger) 모듈
- `tensorrt_int8_calibrator/file_utils.*`
  - 텍스트 및 바이너리 파일 입출력을 돕는 유틸리티 함수 모음

## Outputs

```text
/root/data_with_weight_file/quantized/best_model_int8_calib.cache
/root/data_with_weight_file/quantized/best_model_int8.engine
```
