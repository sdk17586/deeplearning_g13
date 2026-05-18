#include "app_config.hpp"

#include <gst/gst.h>
#include <limits.h>
#include <unistd.h>

#include <filesystem>

namespace {

std::filesystem::path app_root_from_executable() {
  char executable_path[PATH_MAX] = {};
  const ssize_t length = readlink("/proc/self/exe", executable_path,
                                  sizeof(executable_path) - 1);
  if (length <= 0) {
    return std::filesystem::current_path();
  }

  executable_path[length] = '\0';
  std::filesystem::path binary_dir =
      std::filesystem::path(executable_path).parent_path();
  if (binary_dir.filename() == "build") {
    return binary_dir.parent_path();
  }
  return binary_dir;
}

}  // namespace

void AppConfig::SetDefaultConfigPaths(AppConfig &config) {
  const std::filesystem::path config_dir = app_root_from_executable() / "configs";
  config.infer_config = (config_dir / "config_infer_primary_action.txt").string();
  config.preprocess_config = (config_dir / "config_preprocess_action.txt").string();
}

std::string AppConfig::ToUri(const char *input) {
  if (g_str_has_prefix(input, "file://") || g_str_has_prefix(input, "rtsp://") ||
      g_str_has_prefix(input, "http://") || g_str_has_prefix(input, "https://")) {
    return input;
  }

  gchar *absolute = g_canonicalize_filename(input, nullptr);
  gchar *uri = gst_filename_to_uri(absolute, nullptr);
  std::string result = uri ? uri : input;
  g_free(uri);
  g_free(absolute);
  return result;
}

bool AppConfig::FromArgs(int argc, char **argv, AppConfig &config) {
  SetDefaultConfigPaths(config);

  if (argc < 2) {
    g_printerr("Usage: %s <video path|uri> [infer_config] [preprocess_config]\n", argv[0]);
    return false;
  }

  config.uri = ToUri(argv[1]);
  if (argc > 2) {
    config.infer_config = argv[2];
  }
  if (argc > 3) {
    config.preprocess_config = argv[3];
  }
  return true;
}
