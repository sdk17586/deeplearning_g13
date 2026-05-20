#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${APP_DIR}/build/deepstream_action_predict"
DATA_DIR="${APP_DIR}/../../data_with_weight_file"
DEFAULT_VIDEO="${DATA_DIR}/dataset/val/clips/broken/broken_0000.mp4"
DEFAULT_VAL_DIR="${DATA_DIR}/dataset/val"
LOG_FILTER='(^INFO:|[[:space:]]INFO[[:space:]]|INFO: CUSTOM_LIB|Implicit Engine Info|NvDsInferContext|notifyLoadModelStatus|sequence_image_process\.cpp|Opening in BLOCKING MODE|nvstreammux: Successfully handled EOS|Unknown or legacy key specified)'

export LANG="C.utf8"
export LC_ALL="C.utf8"
export LC_CTYPE="C.utf8"

if [[ ! -x "${APP}" ]]; then
  "${APP_DIR}/build.sh"
fi

normalize_label() {
  case "$1" in
    brocken) printf 'broken\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

model_precision_from_config() {
  local infer_config="${1:-${APP_DIR}/configs/config_infer_primary_action.txt}"
  local engine=""
  local network_mode=""

  if [[ -f "${infer_config}" ]]; then
    engine="$(awk -F= '/^[[:space:]]*model-engine-file[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print tolower($2); exit}' "${infer_config}")"
    network_mode="$(awk -F= '/^[[:space:]]*network-mode[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2; exit}' "${infer_config}")"
  fi

  case "${engine}" in
    *int8*) printf 'INT8\n'; return ;;
    *fp16*) printf 'FP16\n'; return ;;
    *fp32*) printf 'FP32\n'; return ;;
  esac

  case "${network_mode}" in
    0) printf 'FP32\n' ;;
    1) printf 'INT8\n' ;;
    2) printf 'FP16\n' ;;
    *) printf 'unknown\n' ;;
  esac
}

start_resource_monitor() {
  RESOURCE_LOG=""
  RESOURCE_MONITOR_PID=""

  if ! command -v tegrastats >/dev/null 2>&1; then
    return
  fi

  RESOURCE_LOG="$(mktemp /tmp/deepstream_action_tegrastats.XXXXXX.log)"
  tegrastats --interval 1000 >"${RESOURCE_LOG}" 2>/dev/null &
  RESOURCE_MONITOR_PID="$!"
}

stop_resource_monitor() {
  if [[ -n "${RESOURCE_MONITOR_PID:-}" ]]; then
    kill "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
    wait "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
    RESOURCE_MONITOR_PID=""
  fi
}

print_resource_summary() {
  if [[ -z "${RESOURCE_LOG:-}" || ! -s "${RESOURCE_LOG}" ]]; then
    echo "평균 CPU 사용률: unavailable"
    echo "평균 GPU 사용률: unavailable"
    echo "평균 메모리 사용량: unavailable"
    echo "평균 전력 사용량: unavailable"
    return
  fi

  python3 - "${RESOURCE_LOG}" <<'PY'
import re
import sys

log_path = sys.argv[1]
samples = 0
cpu_sum = 0.0
cpu_samples = 0
gpu_sum = 0.0
gpu_samples = 0
ram_sum = 0.0
ram_total = None
ram_samples = 0
power_sum = 0.0
power_samples = 0

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        samples += 1

        ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if ram:
            ram_sum += int(ram.group(1))
            ram_total = int(ram.group(2))
            ram_samples += 1

        cpu = re.search(r"CPU\s+\[([^\]]+)\]", line)
        if cpu:
            values = [int(v) for v in re.findall(r"(\d+)%", cpu.group(1))]
            if values:
                cpu_sum += sum(values) / len(values)
                cpu_samples += 1

        gpu = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if gpu:
            gpu_sum += int(gpu.group(1))
            gpu_samples += 1

        power = re.search(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)mW", line)
        if power:
            power_sum += int(power.group(1))
            power_samples += 1

def avg(total, count):
    return total / count if count else None

cpu_avg = avg(cpu_sum, cpu_samples)
gpu_avg = avg(gpu_sum, gpu_samples)
ram_avg = avg(ram_sum, ram_samples)
power_avg = avg(power_sum, power_samples)

print(f"리소스 샘플 수: {samples}")
print(f"평균 CPU 사용률: {cpu_avg:.2f}%" if cpu_avg is not None else "평균 CPU 사용률: unavailable")
print(f"평균 GPU 사용률: {gpu_avg:.2f}%" if gpu_avg is not None else "평균 GPU 사용률: unavailable")
if ram_avg is not None:
    suffix = f"/{ram_total}MB" if ram_total is not None else "MB"
    print(f"평균 메모리 사용량: {ram_avg:.0f}{suffix}")
else:
    print("평균 메모리 사용량: unavailable")
print(f"평균 전력 사용량: {power_avg:.0f}mW" if power_avg is not None else "평균 전력 사용량: unavailable")
print(f"리소스 로그: {log_path}")
PY
}

xml_for_video() {
  local val_dir="$1"
  local label="$2"
  local video="$3"
  local stem
  stem="$(basename "${video}")"
  stem="${stem%.*}"

  printf '%s/labels/%s/%s.xml\n' "${val_dir}" "${label}" "${stem}"
}

load_xml_ranges() {
  local xml="$1"
  local label="$2"
  RANGE_STARTS=()
  RANGE_ENDS=()

  [[ -f "${xml}" ]] || return 1

  local pending_start=""
  while read -r kind frame; do
    if [[ "${kind}" == "S" ]]; then
      pending_start="${frame}"
    elif [[ "${kind}" == "E" && -n "${pending_start}" ]]; then
      RANGE_STARTS+=("${pending_start}")
      RANGE_ENDS+=("${frame}")
      pending_start=""
    fi
  done < <(
    awk -v cls="${label}" '
      function attr(line, name,    pat, start, rest, stop) {
        pat = name "=\""
        start = index(line, pat)
        if (start == 0) return ""
        start += length(pat)
        rest = substr(line, start)
        stop = index(rest, "\"")
        if (stop == 0) return ""
        return substr(rest, 1, stop - 1)
      }
      /<track / {
        current_label = attr($0, "label")
      }
      /<(box|points|polygon) / {
        frame = attr($0, "frame")
        outside = attr($0, "outside")
        if (frame == "" || outside != "0") next
        if (current_label == cls "_start") print "S", frame
        else if (current_label == cls "_end") print "E", frame
      }
    ' "${xml}" | sort -k2,2n
  )

  [[ ${#RANGE_STARTS[@]} -gt 0 ]]
}

label_for_frame() {
  local frame="$1"
  local event_label="$2"
  local frame_num=$((10#${frame}))

  if [[ "${event_label}" == "normal" || ${#RANGE_STARTS[@]} -eq 0 ]]; then
    printf 'unlabeled\n'
    return
  fi

  local i
  for i in "${!RANGE_STARTS[@]}"; do
    if (( frame_num >= 10#${RANGE_STARTS[i]} && frame_num <= 10#${RANGE_ENDS[i]} )); then
      printf '%s\n' "${event_label}"
      return
    fi
  done

  printf 'unlabeled\n'
}

run_one_video() {
  local video="$1"
  local event_label="$2"
  local xml="$3"
  shift
  shift
  shift

  if [[ "${event_label}" != "normal" ]]; then
    if load_xml_ranges "${xml}" "${event_label}"; then
      :
    else
      echo "[검증 경고] XML 이벤트 구간을 찾지 못했습니다. 모든 프레임을 스킵합니다: ${xml}" >&2
      RANGE_STARTS=()
      RANGE_ENDS=()
    fi
  else
    RANGE_STARTS=()
    RANGE_ENDS=()
  fi

  set +e
  local output
  output="$("${APP}" "${video}" "$@" 2>&1)"
  local status=$?
  set -e

  local filtered
  filtered="$(printf '%s\n' "${output}" | grep -v -E "${LOG_FILTER}" || true)"

  local frame_total=0
  local frame_correct=0
  local frame_skipped=0
  local last_prediction=""

  while IFS= read -r line; do
    if [[ "${line}" =~ 프레임=([0-9]+).*결과=.*\(([a-z_]+)\) ]]; then
      local frame="${BASH_REMATCH[1]}"
      local prediction="${BASH_REMATCH[2]}"
      local expected
      expected="$(label_for_frame "${frame}" "${event_label}")"
      local result="MISS"
      last_prediction="${prediction}"

      if [[ "${expected}" == "unlabeled" ]]; then
        frame_skipped=$((frame_skipped + 1))
        printf '    frame=%s 정답=unlabeled 예측=%s SKIP\n' \
          "${frame}" "${prediction}" >&2
        continue
      fi

      frame_total=$((frame_total + 1))
      if [[ "${prediction}" == "${expected}" ]]; then
        frame_correct=$((frame_correct + 1))
        result="OK"
      fi

      printf '    frame=%s 정답=%s 예측=%s %s\n' \
        "${frame}" "${expected}" "${prediction}" "${result}" >&2
    fi
  done <<< "${filtered}"

  if [[ ${frame_total} -eq 0 && ${frame_skipped} -eq 0 && ${status} -ne 0 ]]; then
    printf '%s\n' "${filtered}" | tail -n 12 >&2
  fi

  printf '%s %s %s %s\n' "${frame_total}" "${frame_correct}" "${frame_skipped}" "${last_prediction:-none}"
  return "${status}"
}

run_validation_by_process() {
  local val_dir="$1"
  shift
  local clips_dir="${val_dir}/clips"
  local infer_config="${1:-${APP_DIR}/configs/config_infer_primary_action.txt}"
  local model_precision
  model_precision="$(model_precision_from_config "${infer_config}")"
  local start_ns
  start_ns="$(date +%s%N)"

  if [[ ! -d "${clips_dir}" ]]; then
    echo "[검증 오류] clips 디렉터리가 없습니다: ${clips_dir}" >&2
    return 1
  fi

  echo
  echo "========== 검증 시작 =========="
  echo "검증 데이터셋: ${val_dir}"
  echo "모델 정밀도: ${model_precision}"
  echo "평가 방식: XML 이벤트 구간만 추론/평가"
  echo "================================"
  echo

  local total=0
  local correct=0
  local failed=0
  local frame_total_all=0
  local frame_correct_all=0
  local frame_skipped_all=0
  local inference_time_ns_all=0

  start_resource_monitor
  trap 'stop_resource_monitor' EXIT
  trap 'stop_resource_monitor; exit 130' INT TERM

  while IFS= read -r -d '' video; do
    local expected
    expected="$(normalize_label "$(basename "$(dirname "${video}")")")"
    local xml
    xml="$(xml_for_video "${val_dir}" "${expected}" "${video}")"
    total=$((total + 1))

    local stats=""
    local video_start_ns
    local video_end_ns
    video_start_ns="$(date +%s%N)"
    if stats="$(run_one_video "${video}" "${expected}" "${xml}" "$@")"; then
      :
    else
      failed=$((failed + 1))
    fi
    video_end_ns="$(date +%s%N)"

    local frame_total frame_correct frame_skipped predicted
    read -r frame_total frame_correct frame_skipped predicted <<< "${stats:-0 0 0 none}"
    frame_total_all=$((frame_total_all + frame_total))
    frame_correct_all=$((frame_correct_all + frame_correct))
    frame_skipped_all=$((frame_skipped_all + frame_skipped))
    inference_time_ns_all=$((inference_time_ns_all + video_end_ns - video_start_ns))

    local result="MISS"
    if [[ ${frame_total} -gt 0 && ${frame_correct} -eq ${frame_total} ]]; then
      correct=$((correct + 1))
      result="OK"
    elif [[ ${frame_total} -eq 0 ]]; then
      predicted="none"
    fi

    printf '[%d] %s  이벤트=%s  마지막예측=%s  이벤트프레임정답=%d/%d  스킵=%d  %s\n' \
      "${total}" "$(basename "${video}")" "${expected}" "${predicted}" \
      "${frame_correct}" "${frame_total}" "${frame_skipped}" "${result}"
  done < <(find "${clips_dir}" -mindepth 2 -maxdepth 2 \( -type f -o -type l \) -name '*.mp4' -print0 | sort -z)

  local accuracy="0.00"
  if [[ ${frame_total_all} -gt 0 ]]; then
    accuracy="$(awk -v c="${frame_correct_all}" -v t="${frame_total_all}" 'BEGIN { printf "%.2f", c * 100 / t }')"
  fi
  local end_ns
  end_ns="$(date +%s%N)"
  local total_time_sec
  local inference_time_sec
  local fps
  local avg_infer_ms
  total_time_sec="$(awk -v ns="$((end_ns - start_ns))" 'BEGIN { printf "%.3f", ns / 1000000000 }')"
  inference_time_sec="$(awk -v ns="${inference_time_ns_all}" 'BEGIN { printf "%.3f", ns / 1000000000 }')"
  fps="$(awk -v n="${frame_total_all}" -v ns="${inference_time_ns_all}" 'BEGIN { if (ns > 0) printf "%.2f", n / (ns / 1000000000); else printf "0.00" }')"
  avg_infer_ms="$(awk -v n="${frame_total_all}" -v ns="${inference_time_ns_all}" 'BEGIN { if (n > 0) printf "%.2f", (ns / 1000000) / n; else printf "0.00" }')"

  stop_resource_monitor
  trap - EXIT INT TERM

  echo
  echo "========== 검증 결과 =========="
  echo "이벤트 프레임 전체 정답 영상 수: ${correct}/${total}"
  echo "이벤트 프레임/윈도우 정답 수: ${frame_correct_all}/${frame_total_all}"
  echo "스킵한 unlabeled 프레임/윈도우 수: ${frame_skipped_all}"
  echo "실패/예측없음: ${failed}"
  echo "Event-only Accuracy: ${accuracy}%"
  echo "모델 정밀도: ${model_precision}"
  echo "총 실행 시간: ${total_time_sec}초"
  echo "추론 실행 시간: ${inference_time_sec}초"
  echo "평가 윈도우 처리 FPS: ${fps}"
  echo "윈도우당 평균 추론 시간: ${avg_infer_ms}ms"
  print_resource_summary
  echo "================================"

  [[ ${total} -gt 0 ]]
}

if [[ $# -lt 1 ]]; then
  if [[ -d "${DEFAULT_VAL_DIR}" ]]; then
    echo "[안내] 입력 인자가 없어 기본 검증 데이터셋을 사용합니다."
    echo "[안내] ${DEFAULT_VAL_DIR}"
    set -- --validate "${DEFAULT_VAL_DIR}"
  elif [[ ! -f "${DEFAULT_VIDEO}" ]]; then
    echo "Usage: $0 <video path|uri> [infer_config] [preprocess_config]" >&2
    echo "       $0 --validate <val dataset dir> [infer_config] [preprocess_config]" >&2
    echo "Default validation dataset not found: ${DEFAULT_VAL_DIR}" >&2
    echo "Default video not found: ${DEFAULT_VIDEO}" >&2
    exit 1
  else
    echo "[안내] 기본 검증 데이터셋이 없어 기본 샘플을 사용합니다."
    echo "[안내] ${DEFAULT_VIDEO}"
    set -- "${DEFAULT_VIDEO}"
  fi
fi

cd "${APP_DIR}"
export GST_DEBUG="${GST_DEBUG:-1}"

if [[ "${1:-}" == "--validate" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Usage: $0 --validate <val dataset dir> [infer_config] [preprocess_config]" >&2
    exit 1
  fi
  val_dir="$2"
  shift 2
  run_validation_by_process "${val_dir}" "$@"
  exit $?
fi

set +e
set +o pipefail
"${APP}" "$@" 2>&1 | grep -v -E \
  "${LOG_FILTER}"
status=${PIPESTATUS[0]}
set -e
set -o pipefail
exit "${status}"
