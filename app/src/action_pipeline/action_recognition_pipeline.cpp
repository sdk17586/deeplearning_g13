#include "action_recognition_pipeline.hpp"

#include <cstdlib>
#include <utility>

#include "video_input_source_bin.hpp"

ActionRecognitionPipeline::ActionRecognitionPipeline(AppConfig config)
    : config_(std::move(config)) {}

ActionRecognitionPipeline::~ActionRecognitionPipeline() {
  Cleanup();
}

gboolean ActionRecognitionPipeline::BusCall(GstBus *, GstMessage *msg, gpointer data) {
  auto *self = static_cast<ActionRecognitionPipeline *>(data);
  if (!self || !self->loop_) {
    return TRUE;
  }

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_print("[상태] 영상 처리가 완료되었습니다.\n");
      g_main_loop_quit(self->loop_);
      break;
    case GST_MESSAGE_WARNING: {
      GError *error = nullptr;
      gchar *debug = nullptr;
      gst_message_parse_warning(msg, &error, &debug);
      g_printerr("[경고] %s: %s\n", GST_OBJECT_NAME(msg->src),
                 error ? error->message : "unknown");
      if (debug) {
        g_printerr("[경고 상세] %s\n", debug);
      }
      g_clear_error(&error);
      g_free(debug);
      break;
    }
    case GST_MESSAGE_ERROR: {
      GError *error = nullptr;
      gchar *debug = nullptr;
      gst_message_parse_error(msg, &error, &debug);
      g_printerr("[오류] %s: %s\n", GST_OBJECT_NAME(msg->src),
                 error ? error->message : "unknown");
      if (debug) {
        g_printerr("[오류 상세] %s\n", debug);
      }
      g_clear_error(&error);
      g_free(debug);
      g_main_loop_quit(self->loop_);
      break;
    }
    default:
      break;
  }

  return TRUE;
}

bool ActionRecognitionPipeline::CreateElements() {
  loop_ = g_main_loop_new(nullptr, FALSE);
  pipeline_ = gst_pipeline_new("action-predict-pipeline");

  VideoInputSourceBin source_bin(0, config_.uri);
  source_ = source_bin.Create();

  streammux_ = gst_element_factory_make("nvstreammux", "stream-muxer");
  queue_preprocess_ = gst_element_factory_make("queue", "queue-preprocess");
  preprocess_ = gst_element_factory_make("nvdspreprocess", "preprocess");
  queue_infer_ = gst_element_factory_make("queue", "queue-infer");
  infer_ = gst_element_factory_make("nvinfer", "primary-infer");
  queue_sink_ = gst_element_factory_make("queue", "queue-sink");
  sink_ = gst_element_factory_make("fakesink", "sink");

  if (!loop_ || !pipeline_ || !source_ || !streammux_ || !queue_preprocess_ ||
      !preprocess_ || !queue_infer_ || !infer_ || !queue_sink_ || !sink_) {
    g_printerr("Failed to create one or more GStreamer/DeepStream elements\n");
    return false;
  }

  return true;
}

bool ActionRecognitionPipeline::ConfigureElements() {
  g_object_set(G_OBJECT(streammux_), "batch-size", 1, "width", config_.mux_width,
               "height", config_.mux_height, "batched-push-timeout", 40000,
               nullptr);
  g_object_set(G_OBJECT(preprocess_), "config-file",
               config_.preprocess_config.c_str(), nullptr);
  g_object_set(G_OBJECT(infer_), "config-file-path", config_.infer_config.c_str(),
               "input-tensor-meta", TRUE, nullptr);
  g_object_set(G_OBJECT(sink_), "sync", FALSE, "qos", FALSE, nullptr);
  return true;
}

bool ActionRecognitionPipeline::LinkElements() {
  gst_bin_add_many(GST_BIN(pipeline_), source_, streammux_, queue_preprocess_,
                   preprocess_, queue_infer_, infer_, queue_sink_, sink_, nullptr);

  GstPad *streammux_sink_pad = gst_element_request_pad_simple(streammux_, "sink_0");
  GstPad *source_src_pad = gst_element_get_static_pad(source_, "src");
  if (!streammux_sink_pad || !source_src_pad ||
      gst_pad_link(source_src_pad, streammux_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr("Failed to link source to nvstreammux\n");
    if (source_src_pad) {
      gst_object_unref(source_src_pad);
    }
    if (streammux_sink_pad) {
      gst_object_unref(streammux_sink_pad);
    }
    return false;
  }

  gst_object_unref(source_src_pad);
  gst_object_unref(streammux_sink_pad);

  if (!gst_element_link_many(streammux_, queue_preprocess_, preprocess_, queue_infer_,
                             infer_, queue_sink_, sink_, nullptr)) {
    g_printerr("Failed to link pipeline elements\n");
    return false;
  }

  return true;
}

bool ActionRecognitionPipeline::AttachProbes() {
  GstPad *infer_src_pad = gst_element_get_static_pad(infer_, "src");
  if (!infer_src_pad) {
    g_printerr("Unable to get nvinfer src pad\n");
    return false;
  }

  gst_pad_add_probe(infer_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                    PredictionResultHandler::OnInferSrcPadBuffer,
                    &prediction_handler_, nullptr);
  gst_object_unref(infer_src_pad);

  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
  bus_watch_id_ = gst_bus_add_watch(bus, ActionRecognitionPipeline::BusCall, this);
  gst_object_unref(bus);

  return true;
}

void ActionRecognitionPipeline::PrintStartMessage() const {
  g_print("\n========== DeepStream 예측 시작 ==========\n");
  g_print("입력 영상: %s\n", config_.uri.c_str());
  g_print("추론 설정: %s\n", config_.infer_config.c_str());
  g_print("전처리 설정: %s\n", config_.preprocess_config.c_str());
  g_print("==========================================\n\n");
}

int ActionRecognitionPipeline::Run() {
  if (!CreateElements() || !ConfigureElements() || !LinkElements() || !AttachProbes()) {
    return EXIT_FAILURE;
  }

  prediction_handler_.Reset();
  PrintStartMessage();

  gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  g_main_loop_run(loop_);

  prediction_handler_.PrintSummary();
  return EXIT_SUCCESS;
}

void ActionRecognitionPipeline::Cleanup() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
  }

  if (bus_watch_id_ != 0) {
    g_source_remove(bus_watch_id_);
    bus_watch_id_ = 0;
  }

  if (pipeline_) {
    gst_object_unref(GST_OBJECT(pipeline_));
    pipeline_ = nullptr;
  }

  if (loop_) {
    g_main_loop_unref(loop_);
    loop_ = nullptr;
  }

  source_ = nullptr;
  streammux_ = nullptr;
  queue_preprocess_ = nullptr;
  preprocess_ = nullptr;
  queue_infer_ = nullptr;
  infer_ = nullptr;
  queue_sink_ = nullptr;
  sink_ = nullptr;
}
