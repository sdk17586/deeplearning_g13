#pragma once

#include <gst/gst.h>

void reset_prediction_summary();
void print_prediction_summary();

GstPadProbeReturn infer_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                                             gpointer user_data);
