#pragma once

#include "app_config.hpp"
#include "prediction_result_handler.hpp"

#include <gst/gst.h>

#include <optional>
#include <string>

class ActionRecognitionPipeline {
 public:
  explicit ActionRecognitionPipeline(AppConfig config);
  ~ActionRecognitionPipeline();

  ActionRecognitionPipeline(const ActionRecognitionPipeline &) = delete;
  ActionRecognitionPipeline &operator=(const ActionRecognitionPipeline &) = delete;

  int Run();
  std::optional<std::string> PredictedLabelId() const;

 private:
  bool CreateElements();
  bool ConfigureElements();
  bool LinkElements();
  bool AttachProbes();
  void PrintStartMessage() const;
  void Cleanup();

  static gboolean BusCall(GstBus *bus, GstMessage *msg, gpointer data);

  AppConfig config_;
  PredictionResultHandler prediction_handler_;

  GMainLoop *loop_ = nullptr;
  GstElement *pipeline_ = nullptr;
  GstElement *source_ = nullptr;
  GstElement *streammux_ = nullptr;
  GstElement *queue_preprocess_ = nullptr;
  GstElement *preprocess_ = nullptr;
  GstElement *queue_infer_ = nullptr;
  GstElement *infer_ = nullptr;
  GstElement *queue_sink_ = nullptr;
  GstElement *sink_ = nullptr;
  guint bus_watch_id_ = 0;
};
