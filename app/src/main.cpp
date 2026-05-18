#include <clocale>
#include <cstdlib>

#include <gst/gst.h>

#include "app_config.hpp"
#include "pipeline.hpp"

int main(int argc, char *argv[]) {
  std::setlocale(LC_ALL, "");

  gst_init(&argc, &argv);
  gst_debug_set_default_threshold(GST_LEVEL_ERROR);

  AppConfig config;
  if (!parse_args(argc, argv, config)) {
    return EXIT_FAILURE;
  }

  return run_pipeline(config);
}
