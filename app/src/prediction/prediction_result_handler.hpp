#pragma once

#include <gst/gst.h>

#include <array>
#include <vector>

#include "gstnvdsinfer.h"

class PredictionResultHandler {
 public:
  void Reset();
  void PrintSummary() const;

  GstPadProbeReturn HandleBuffer(GstPadProbeInfo *info);

  static GstPadProbeReturn OnInferSrcPadBuffer(GstPad *pad, GstPadProbeInfo *info,
                                               gpointer user_data);

 private:
  struct LabelInfo {
    const char *id;
    const char *name;
  };

  struct PredictionSummary {
    std::array<unsigned int, 5> counts = {};
    std::array<float, 5> confidence_sums = {};
    size_t last_class = 0;
    float last_confidence = 0.0f;
    guint64 last_frame = 0;
    double last_time_sec = 0.0;
    bool has_last_time = false;
    unsigned int total = 0;
  };

  static constexpr std::array<LabelInfo, 5> kLabels = {{
      {"abandon", "방치"},
      {"fight", "싸움"},
      {"broken", "파손"},
      {"theft", "절도"},
      {"normal", "정상"},
  }};

  static std::vector<float> Softmax(const float *values, size_t count);
  static bool PtsToSeconds(guint64 pts, double &seconds);

  void PrintPrediction(const NvDsInferTensorMeta *tensor_meta, guint64 frame_num,
                       guint64 pts);

  PredictionSummary summary_;
};
