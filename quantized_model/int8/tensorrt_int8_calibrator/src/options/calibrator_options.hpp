#pragma once

#include <string>

struct CalibratorOptions {
  std::string onnx = "/home/gitsn_s/data_with_weight_file/quantized/best_model.onnx";
  std::string engine = "/home/gitsn_s/data_with_weight_file/quantized/best_model_int8.engine";
  std::string cache = "/home/gitsn_s/data_with_weight_file/quantized/best_model_int8_calib.cache";
  std::string tensorList = "/home/gitsn_s/data_with_weight_file/quantized/calib_tensors/list.txt";
  int batchSize = 1;
  int channels = 3;
  int frames = 16;
  int height = 112;
  int width = 112;
  int workspaceMiB = 4096;

  static CalibratorOptions parse(int argc, char** argv);
};
