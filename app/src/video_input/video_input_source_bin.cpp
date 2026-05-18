#include "video_input_source_bin.hpp"

#include <utility>

#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

VideoInputSourceBin::VideoInputSourceBin(guint index, std::string uri)
    : index_(index), uri_(std::move(uri)) {}

void VideoInputSourceBin::OnDecodebinChildAdded(GstChildProxy *child_proxy, GObject *object,
                                                gchar *name, gpointer user_data) {
  (void)child_proxy;
  if (g_strrstr(name, "decodebin") == name) {
    g_signal_connect(G_OBJECT(object), "child-added",
                     G_CALLBACK(VideoInputSourceBin::OnDecodebinChildAdded), user_data);
  }
}

void VideoInputSourceBin::OnPadAdded(GstElement *, GstPad *decoder_src_pad, gpointer data) {
  GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
  if (!caps) {
    caps = gst_pad_query_caps(decoder_src_pad, nullptr);
  }
  if (!caps) {
    g_printerr("Could not get decodebin pad caps\n");
    return;
  }

  const GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);
  GstCapsFeatures *features = gst_caps_get_features(caps, 0);

  if (g_str_has_prefix(name, "video")) {
    if (!gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM)) {
      g_printerr("Decodebin did not select an NVIDIA decoder with NVMM output.\n");
      gst_caps_unref(caps);
      return;
    }

    auto *source_bin = static_cast<GstElement *>(data);
    GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
    if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad), decoder_src_pad)) {
      g_printerr("Failed to link decodebin src pad to source bin ghost pad\n");
    }
    gst_object_unref(bin_ghost_pad);
  }

  gst_caps_unref(caps);
}

GstElement *VideoInputSourceBin::Create() const {
  gchar bin_name[32] = {};
  g_snprintf(bin_name, sizeof(bin_name), "source-bin-%u", index_);

  GstElement *bin = gst_bin_new(bin_name);
  GstElement *uri_decode_bin = gst_element_factory_make("uridecodebin", nullptr);
  if (!bin || !uri_decode_bin) {
    g_printerr("Failed to create source bin elements\n");
    return nullptr;
  }

  g_object_set(G_OBJECT(uri_decode_bin), "uri", uri_.c_str(), nullptr);
  g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added",
                   G_CALLBACK(VideoInputSourceBin::OnPadAdded), bin);
  g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                   G_CALLBACK(VideoInputSourceBin::OnDecodebinChildAdded), bin);

  gst_bin_add(GST_BIN(bin), uri_decode_bin);
  if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src", GST_PAD_SRC))) {
    g_printerr("Failed to add source bin ghost pad\n");
    return nullptr;
  }

  return bin;
}
