#pragma once

#include <glib.h>

#include <string>

struct AppConfig {
  std::string uri;
  std::string infer_config;
  std::string preprocess_config;
  guint mux_width = 1280;
  guint mux_height = 720;
};

bool parse_args(int argc, char **argv, AppConfig &config);
