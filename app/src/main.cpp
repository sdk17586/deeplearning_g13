#include <clocale>
#include <cstdlib>

#include <gst/gst.h>

#include "app_config.hpp"
#include "action_recognition_pipeline.hpp"
#include "validation_runner.hpp"

int main(int argc, char *argv[]) {
  std::setlocale(LC_ALL, "");

  gst_init(&argc, &argv);
  gst_debug_set_default_threshold(GST_LEVEL_ERROR);

  AppConfig config;
  if (!AppConfig::FromArgs(argc, argv, config)) {
    return EXIT_FAILURE;
  }

  if (config.mode == AppConfig::Mode::ValidateDataset) {
    return RunValidation(config);
  }

  ActionRecognitionPipeline pipeline(config);
  return pipeline.Run();
}
