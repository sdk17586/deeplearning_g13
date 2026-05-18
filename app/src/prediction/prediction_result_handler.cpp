#include "prediction_result_handler.hpp"

#include <algorithm>
#include <cmath>

#include "gstnvdsmeta.h"
#include "nvdsinfer.h"
#include "nvdsmeta.h"
#include "nvdspreprocess_meta.h"

std::vector<float> PredictionResultHandler::Softmax(const float *values, size_t count) {
  std::vector<float> probs(count);
  if (count == 0) {
    return probs;
  }

  float max_value = *std::max_element(values, values + count);
  float sum = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    probs[i] = std::exp(values[i] - max_value);
    sum += probs[i];
  }

  if (sum > 0.0f) {
    for (float &prob : probs) {
      prob /= sum;
    }
  }
  return probs;
}

bool PredictionResultHandler::PtsToSeconds(guint64 pts, double &seconds) {
  if (pts == GST_CLOCK_TIME_NONE) {
    return false;
  }
  seconds = static_cast<double>(pts) / static_cast<double>(GST_SECOND);
  return true;
}

void PredictionResultHandler::PrintPrediction(const NvDsInferTensorMeta *tensor_meta,
                                              guint64 frame_num, guint64 pts) {
  if (!tensor_meta || tensor_meta->num_output_layers < 1 ||
      !tensor_meta->out_buf_ptrs_host[0]) {
    return;
  }

  const NvDsInferLayerInfo &layer = tensor_meta->output_layers_info[0];
  const size_t count = std::min<size_t>(layer.inferDims.numElements, kLabels.size());
  if (count == 0 || layer.dataType != NvDsInferDataType::FLOAT) {
    g_printerr("Unsupported output tensor format or empty tensor\n");
    return;
  }

  const auto *logits = static_cast<const float *>(tensor_meta->out_buf_ptrs_host[0]);
  std::vector<float> probs = Softmax(logits, count);
  const auto best_iter = std::max_element(probs.begin(), probs.end());
  const size_t best_idx = std::distance(probs.begin(), best_iter);
  const float confidence = probs[best_idx] * 100.0f;

  summary_.counts[best_idx]++;
  summary_.confidence_sums[best_idx] += confidence;
  summary_.last_class = best_idx;
  summary_.last_confidence = confidence;
  summary_.last_frame = frame_num;
  summary_.has_last_time = PtsToSeconds(pts, summary_.last_time_sec);
  summary_.total++;

  double time_sec = 0.0;
  if (PtsToSeconds(pts, time_sec)) {
    g_print("[예측] 시간=%6.2f초  프레임=%03" G_GUINT64_FORMAT
            "  결과=%s(%s)  신뢰도=%.2f%%\n",
            time_sec, frame_num, kLabels[best_idx].name, kLabels[best_idx].id,
            confidence);
  } else {
    g_print("[예측] 시간=알수없음  프레임=%03" G_GUINT64_FORMAT
            "  결과=%s(%s)  신뢰도=%.2f%%\n",
            frame_num, kLabels[best_idx].name, kLabels[best_idx].id, confidence);
  }
}

void PredictionResultHandler::Reset() {
  summary_ = PredictionSummary{};
}

void PredictionResultHandler::PrintSummary() const {
  g_print("\n========== 예측 요약 ==========\n");

  if (summary_.total == 0) {
    g_print("예측 결과가 없습니다.\n");
    g_print("================================\n");
    return;
  }

  size_t majority_class = 0;
  for (size_t i = 1; i < kLabels.size(); ++i) {
    if (summary_.counts[i] > summary_.counts[majority_class]) {
      majority_class = i;
    }
  }

  g_print("총 예측 횟수: %u\n", summary_.total);
  if (summary_.has_last_time) {
    g_print("최종 예측: %s(%s), 신뢰도 %.2f%%, 마지막 시간 %.2f초, 마지막 프레임 %"
            G_GUINT64_FORMAT "\n",
            kLabels[summary_.last_class].name, kLabels[summary_.last_class].id,
            summary_.last_confidence, summary_.last_time_sec, summary_.last_frame);
  } else {
    g_print("최종 예측: %s(%s), 신뢰도 %.2f%%, 마지막 프레임 %" G_GUINT64_FORMAT "\n",
            kLabels[summary_.last_class].name, kLabels[summary_.last_class].id,
            summary_.last_confidence, summary_.last_frame);
  }
  g_print("다수결 예측: %s(%s), %u/%u회\n",
          kLabels[majority_class].name, kLabels[majority_class].id,
          summary_.counts[majority_class], summary_.total);

  g_print("클래스별 집계:\n");
  for (size_t i = 0; i < kLabels.size(); ++i) {
    const float avg_confidence =
        summary_.counts[i] > 0 ? summary_.confidence_sums[i] / summary_.counts[i] : 0.0f;
    g_print("  - %s(%s): %u회, 평균 신뢰도 %.2f%%\n",
            kLabels[i].name, kLabels[i].id, summary_.counts[i], avg_confidence);
  }
  g_print("================================\n");
}

GstPadProbeReturn PredictionResultHandler::OnInferSrcPadBuffer(GstPad *, GstPadProbeInfo *info,
                                                               gpointer user_data) {
  auto *handler = static_cast<PredictionResultHandler *>(user_data);
  return handler ? handler->HandleBuffer(info) : GST_PAD_PROBE_OK;
}

GstPadProbeReturn PredictionResultHandler::HandleBuffer(GstPadProbeInfo *info) {
  auto *buf = static_cast<GstBuffer *>(info->data);
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) {
    return GST_PAD_PROBE_OK;
  }

  guint64 frame_num = 0;
  guint64 pts = GST_CLOCK_TIME_NONE;
  if (batch_meta->frame_meta_list && batch_meta->frame_meta_list->data) {
    auto *frame_meta = static_cast<NvDsFrameMeta *>(batch_meta->frame_meta_list->data);
    frame_num = frame_meta->frame_num;
    pts = frame_meta->buf_pts;
  }

  for (NvDsMetaList *l_user_meta = batch_meta->batch_user_meta_list;
       l_user_meta != nullptr; l_user_meta = l_user_meta->next) {
    auto *user_meta = static_cast<NvDsUserMeta *>(l_user_meta->data);
    if (!user_meta || user_meta->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META) {
      continue;
    }

    auto *preprocess_meta =
        static_cast<GstNvDsPreProcessBatchMeta *>(user_meta->user_meta_data);
    for (auto &roi_meta : preprocess_meta->roi_vector) {
      guint64 roi_frame_num = roi_meta.frame_meta ? roi_meta.frame_meta->frame_num : frame_num;
      guint64 roi_pts = roi_meta.frame_meta ? roi_meta.frame_meta->buf_pts : pts;
      for (NvDsMetaList *l_roi_user = roi_meta.roi_user_meta_list;
           l_roi_user != nullptr; l_roi_user = l_roi_user->next) {
        auto *roi_user_meta = static_cast<NvDsUserMeta *>(l_roi_user->data);
        if (!roi_user_meta ||
            roi_user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) {
          continue;
        }
        auto *tensor_meta =
            static_cast<NvDsInferTensorMeta *>(roi_user_meta->user_meta_data);
        PrintPrediction(tensor_meta, roi_frame_num, roi_pts);
      }
    }
  }

  return GST_PAD_PROBE_OK;
}
