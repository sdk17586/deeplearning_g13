#include "pipeline.hpp"

#include <cstdlib>

#include <gst/gst.h>

#include "infer_probe.hpp"
#include "source_bin.hpp"

namespace {

gboolean bus_call(GstBus *, GstMessage *msg, gpointer data) {
  auto *loop = static_cast<GMainLoop *>(data);

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_print("[상태] 영상 처리가 완료되었습니다.\n");
      g_main_loop_quit(loop);
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
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }

  return TRUE;
}

}  // namespace

int run_pipeline(const AppConfig &config) {
  GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
  GstElement *pipeline = gst_pipeline_new("action-predict-pipeline");
  GstElement *source = create_source_bin(0, config.uri);
  GstElement *streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
  GstElement *queue1 = gst_element_factory_make("queue", "queue-preprocess");
  GstElement *preprocess = gst_element_factory_make("nvdspreprocess", "preprocess");
  GstElement *queue2 = gst_element_factory_make("queue", "queue-infer");
  GstElement *infer = gst_element_factory_make("nvinfer", "primary-infer");
  GstElement *queue3 = gst_element_factory_make("queue", "queue-sink");
  GstElement *sink = gst_element_factory_make("fakesink", "sink");

  if (!loop || !pipeline || !source || !streammux || !queue1 || !preprocess ||
      !queue2 || !infer || !queue3 || !sink) {
    g_printerr("Failed to create one or more GStreamer/DeepStream elements\n");
    return EXIT_FAILURE;
  }

  g_object_set(G_OBJECT(streammux), "batch-size", 1, "width", config.mux_width,
               "height", config.mux_height, "batched-push-timeout", 40000, nullptr);
  g_object_set(G_OBJECT(preprocess), "config-file", config.preprocess_config.c_str(), nullptr);
  g_object_set(G_OBJECT(infer), "config-file-path", config.infer_config.c_str(),
               "input-tensor-meta", TRUE, nullptr);
  g_object_set(G_OBJECT(sink), "sync", FALSE, "qos", FALSE, nullptr);

  gst_bin_add_many(GST_BIN(pipeline), source, streammux, queue1, preprocess,
                   queue2, infer, queue3, sink, nullptr);

  GstPad *streammux_sink_pad = gst_element_request_pad_simple(streammux, "sink_0");
  GstPad *source_src_pad = gst_element_get_static_pad(source, "src");
  if (!streammux_sink_pad || !source_src_pad ||
      gst_pad_link(source_src_pad, streammux_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr("Failed to link source to nvstreammux\n");
    return EXIT_FAILURE;
  }
  gst_object_unref(source_src_pad);
  gst_object_unref(streammux_sink_pad);

  if (!gst_element_link_many(streammux, queue1, preprocess, queue2, infer, queue3,
                             sink, nullptr)) {
    g_printerr("Failed to link pipeline elements\n");
    return EXIT_FAILURE;
  }

  GstPad *infer_src_pad = gst_element_get_static_pad(infer, "src");
  if (!infer_src_pad) {
    g_printerr("Unable to get nvinfer src pad\n");
    return EXIT_FAILURE;
  }
  gst_pad_add_probe(infer_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                    infer_src_pad_buffer_probe, nullptr, nullptr);
  gst_object_unref(infer_src_pad);

  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  guint bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  reset_prediction_summary();

  g_print("\n========== DeepStream 예측 시작 ==========\n");
  g_print("입력 영상: %s\n", config.uri.c_str());
  g_print("추론 설정: %s\n", config.infer_config.c_str());
  g_print("전처리 설정: %s\n", config.preprocess_config.c_str());
  g_print("==========================================\n\n");

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  print_prediction_summary();

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  return EXIT_SUCCESS;
}
