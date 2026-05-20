#pragma once

#include <glib.h>

#include <string>

struct AppConfig {
  enum class Mode {
    PredictOne,
    ValidateDataset,
  };

  Mode mode = Mode::PredictOne;
  std::string uri;
  std::string validation_dir;
  std::string infer_config;
  std::string preprocess_config;
  guint mux_width = 1280;
  guint mux_height = 720;
  bool print_predictions = true;
  bool print_summary = true;

  static bool FromArgs(int argc, char **argv, AppConfig &config);
  static std::string ToUri(const char *input);

 private:
  static void SetDefaultConfigPaths(AppConfig &config);
};
