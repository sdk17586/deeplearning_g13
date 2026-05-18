#pragma once

#include <gst/gst.h>

#include <string>

class VideoInputSourceBin {
 public:
  VideoInputSourceBin(guint index, std::string uri);

  GstElement *Create() const;

 private:
  static void OnDecodebinChildAdded(GstChildProxy *child_proxy, GObject *object,
                                    gchar *name, gpointer user_data);
  static void OnPadAdded(GstElement *decodebin, GstPad *decoder_src_pad,
                         gpointer data);

  guint index_;
  std::string uri_;
};
