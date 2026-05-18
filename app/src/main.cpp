#include <clocale>
#include <cstdlib>

#include <gst/gst.h>

#include "app_config.hpp"
#include "action_recognition_pipeline.hpp"

int main(int argc, char *argv[]) {
  std::setlocale(LC_ALL, "");

  gst_init(&argc, &argv);
  gst_debug_set_default_threshold(GST_LEVEL_ERROR);

  AppConfig config;
  if (!AppConfig::FromArgs(argc, argv, config)) {
    return EXIT_FAILURE;
  }

  ActionRecognitionPipeline pipeline(config);
  return pipeline.Run();
}
