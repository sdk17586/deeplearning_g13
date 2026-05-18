#pragma once

#include <gst/gst.h>

#include <string>

GstElement *create_source_bin(guint index, const std::string &uri);
